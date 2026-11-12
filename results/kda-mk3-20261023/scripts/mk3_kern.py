"""Per-kernel device time of one flash-train KDA forward and backward (Megatron-LM-k3 call by default).
  python3 mk3_kern.py T H [mode mk3|k3]  -> JSON: {"fwd": {kernel: ms}, "bwd": {...}, totals}"""
import json, sys
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
import flash_train.torch as ft

T, H = int(sys.argv[1]), int(sys.argv[2])
MODE = sys.argv[3] if len(sys.argv) > 3 else "mk3"
gen = torch.Generator(device="cuda").manual_seed(T + H)
r = lambda *s: torch.randn(*s, device="cuda", generator=gen)
q0, k0 = r(1, T, H, 128), r(1, T, H, 128)
if MODE == "mk3":
    q0, k0 = F.normalize(q0, dim=-1), F.normalize(k0, dim=-1)
X = {"q": q0.bfloat16(), "k": k0.bfloat16(), "v": (r(1, T, H, 128) * .3).bfloat16(),
     "g": (r(1, T, H, 128) * .3).bfloat16(), "beta": r(1, T, H).sigmoid(),
     "A_log": torch.log(1 + 15 * torch.rand(H, device="cuda", generator=gen)), "dt_bias": r(H * 128) * .1}
L = {k: v.requires_grad_(True) for k, v in X.items()}
D_O = r(1, T, H, 128).bfloat16()
CU = torch.tensor([0, T], dtype=torch.int32, device="cuda")
KW = dict(A_log=L["A_log"], dt_bias=L["dt_bias"], use_qk_l2norm_in_kernel=(MODE == "k3"), use_gate_in_kernel=True,
          safe_gate=True, lower_bound=-5.0, cu_seqlens=CU)
f = lambda: ft.chunk_kda(L["q"], L["k"], L["v"], L["g"], L["beta"], **KW)[0]


def kern(prof):
    out = {}
    for e in prof.key_averages():
        if e.device_type.name == "CUDA" and e.device_time_total > 0:
            out[e.key[:60]] = out.get(e.key[:60], 0) + e.device_time_total / 1e3 / N
    return out


N = 5
for _ in range(3):
    f().backward(D_O)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as pf:
    for _ in range(N):
        o = f()
    torch.cuda.synchronize()
o = f()
with profile(activities=[ProfilerActivity.CUDA]) as pb:
    for _ in range(N):
        for t in L.values(): t.grad = None
        o.backward(D_O, retain_graph=True)
    torch.cuda.synchronize()
fw, bw = kern(pf), kern(pb)
print(json.dumps({"T": T, "H": H, "mode": MODE, "fwd_ms": round(sum(fw.values()), 4), "bwd_ms": round(sum(bw.values()), 4),
                  "fwd": {k: round(v, 4) for k, v in sorted(fw.items(), key=lambda x: -x[1])},
                  "bwd": {k: round(v, 4) for k, v in sorted(bw.items(), key=lambda x: -x[1])}}))
