"""Forward-only per-kernel device time (us per call), FLA (IMPL=fla) or flash-train (IMPL=ft)."""
import os, sys, json, collections, re, torch, torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
impl = os.environ.get("IMPL", "ft")
if impl == "fla":
    from fla.ops.kda import chunk_kda
else:
    from flash_train.torch import chunk_kda
for sh in sys.argv[1].split(","):
    T, H = map(int, sh.split("x"))
    g = torch.Generator(device="cuda").manual_seed(T + H)
    r = lambda *s: torch.randn(*s, device="cuda", generator=g)
    q = F.normalize(r(1, T, H, 128), dim=-1).bfloat16(); k = F.normalize(r(1, T, H, 128), dim=-1).bfloat16()
    v = (r(1, T, H, 128) * .3).bfloat16(); gg = (r(1, T, H, 128) * .3).bfloat16(); beta = r(1, T, H).sigmoid()
    A_log = torch.log(1 + 15 * torch.rand(H, device="cuda", generator=g)); dt = r(H * 128) * .1
    CU = torch.tensor([0, T], dtype=torch.int32 if impl == "ft" else torch.long, device="cuda")
    KW = dict(A_log=A_log, dt_bias=dt, use_qk_l2norm_in_kernel=False, use_gate_in_kernel=True, safe_gate=True,
              lower_bound=-5.0, cu_seqlens=CU)
    with torch.no_grad():
        for _ in range(3): chunk_kda(q, k, v, gg, beta, **KW)
        torch.cuda.synchronize(); N = 10
        with profile(activities=[ProfilerActivity.CUDA]) as pr:
            for _ in range(N): chunk_kda(q, k, v, gg, beta, **KW)
            torch.cuda.synchronize()
    per = collections.Counter()
    for e in pr.events():
        if e.device_type.name == "CUDA":
            per[re.sub(r"<.*|\(.*", "", e.name.replace("void ", "")).split("::")[-1][:48]] += e.device_time / N
    print(json.dumps({"impl": impl, "dev": torch.cuda.get_device_name(), "T": T, "H": H, "tot": round(sum(per.values()), 1),
                      "per": {a: round(b, 1) for a, b in per.most_common(12)}}), flush=True)
