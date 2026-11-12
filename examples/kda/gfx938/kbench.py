"""Fast A/B timer: ONE process per build, every shape timed in-process.
  python3 kbench.py --key WyS4AParams --shapes 2048x16,8192x64 --iters 20 [--tag NAME]
Per shape: 2 warm-up fwd+bwd, then N profiled fwd+bwd; prints JSON with the per-iteration device time of every
kernel whose name contains --key (median, min) and the per-iteration total device time (median, min)."""
import argparse, json, statistics as st, torch, torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
import flash_train.torch as ft
ap = argparse.ArgumentParser()
ap.add_argument("--key", default="")
ap.add_argument("--shapes", default="2048x16,8192x64")
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--tag", default="")
ap.add_argument("--all", action="store_true")
a = ap.parse_args()
for sh in a.shapes.split(","):
    T, H = map(int, sh.split("x"))
    g = torch.Generator(device="cuda").manual_seed(T + H)
    r = lambda *s: torch.randn(*s, device="cuda", generator=g)
    X = {"q": F.normalize(r(1, T, H, 128), dim=-1).bfloat16(), "k": F.normalize(r(1, T, H, 128), dim=-1).bfloat16(),
         "v": (r(1, T, H, 128) * .3).bfloat16(), "g": (r(1, T, H, 128) * .3).bfloat16(), "beta": r(1, T, H).sigmoid(),
         "A_log": torch.log(1 + 15 * torch.rand(H, device="cuda", generator=g)), "dt_bias": r(H * 128) * .1}
    L = {k: v.requires_grad_(True) for k, v in X.items()}
    D = r(1, T, H, 128).bfloat16(); CU = torch.tensor([0, T], dtype=torch.int32, device="cuda")
    KW = dict(A_log=L["A_log"], dt_bias=L["dt_bias"], use_qk_l2norm_in_kernel=False, use_gate_in_kernel=True,
              safe_gate=True, lower_bound=-5.0, cu_seqlens=CU)
    def step():
        for t in L.values(): t.grad = None
        ft.chunk_kda(L["q"], L["k"], L["v"], L["g"], L["beta"], **KW)[0].backward(D)
    for _ in range(2): step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        for _ in range(a.iters): step()
        torch.cuda.synchronize()
    ev = [e for e in pr.events() if e.device_type.name == "CUDA"]
    tot = sum(e.device_time for e in ev) / a.iters
    ks = [e.device_time for e in ev if a.key and a.key in e.name]
    per = len(ks) // a.iters if a.iters else 0
    samples = [sum(ks[i * per:(i + 1) * per]) for i in range(a.iters)] if per else []
    rec = {"tag": a.tag, "T": T, "H": H, "k_med": st.median(samples) if samples else None, "k_min": min(samples) if samples else None, "tot_avg": tot}
    if a.all:
        import collections, re
        per_k = collections.Counter()
        for e in ev: per_k[re.sub(r"<.*|\(.*", "", e.name.replace("void ", "")).split("::")[-1]] += e.device_time / a.iters
        rec["per"] = {k: round(v, 1) for k, v in per_k.most_common(14)}
    print(json.dumps(rec), flush=True)
