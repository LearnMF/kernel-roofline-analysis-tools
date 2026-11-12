"""Benchmark-table measurement: forward and backward separately, flash-train vs FLA (Megatron-LM-k3 call).
  IMPL=ft|fla python3 mk3_fb.py T H -> JSON: fwd/bwd device ms (profiler kernel sum, mean of N), fwd/bwd peak MB
  (peak allocated during the direction minus allocated at its start, MB = 1e6 B)."""
import json, os, sys
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
T, H = int(sys.argv[1]), int(sys.argv[2])
IMPL = os.environ.get("IMPL", "ft")
if IMPL == "ft":
    import flash_train.torch as ft
    kda = ft.chunk_kda
else:
    from fla.ops.kda import chunk_kda as kda
gen = torch.Generator(device="cuda").manual_seed(T + H)
r = lambda *s: torch.randn(*s, device="cuda", generator=gen)
q0, k0 = F.normalize(r(1, T, H, 128), dim=-1), F.normalize(r(1, T, H, 128), dim=-1)
X = {"q": q0.bfloat16(), "k": k0.bfloat16(), "v": (r(1, T, H, 128) * .3).bfloat16(),
     "g": (r(1, T, H, 128) * .3).bfloat16(), "beta": r(1, T, H).sigmoid(),
     "A_log": torch.log(1 + 15 * torch.rand(H, device="cuda", generator=gen)), "dt_bias": r(H * 128) * .1}
L = {k: v.requires_grad_(True) for k, v in X.items()}
D_O = r(1, T, H, 128).bfloat16()
CU = torch.tensor([0, T], dtype=torch.int32, device="cuda")
KW = dict(A_log=L["A_log"], dt_bias=L["dt_bias"], use_qk_l2norm_in_kernel=False, use_gate_in_kernel=True,
          safe_gate=True, lower_bound=-5.0, cu_seqlens=CU)
f = lambda: kda(L["q"], L["k"], L["v"], L["g"], L["beta"], **KW)[0]
def zero():
    for t in L.values(): t.grad = None
N = 5
for _ in range(3):
    zero(); f().backward(D_O)
torch.cuda.synchronize()
# memory, one clean pass per direction
zero(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); b0 = torch.cuda.memory_allocated()
o = f(); torch.cuda.synchronize(); fwd_peak = (torch.cuda.max_memory_allocated() - b0) / 1e6
torch.cuda.reset_peak_memory_stats(); b1 = torch.cuda.memory_allocated()
o.backward(D_O); torch.cuda.synchronize(); bwd_peak = (torch.cuda.max_memory_allocated() - b1) / 1e6
del o
def ksum(prof):
    return sum(e.device_time_total for e in prof.key_averages() if e.device_type.name == "CUDA") / 1e3 / N
with profile(activities=[ProfilerActivity.CUDA]) as pf:
    for _ in range(N): o = f()
    torch.cuda.synchronize()
o = f()
with profile(activities=[ProfilerActivity.CUDA]) as pb:
    for _ in range(N):
        zero(); o.backward(D_O, retain_graph=True)
    torch.cuda.synchronize()
print(json.dumps({"impl": IMPL, "arch": torch.cuda.get_device_properties(0).gcnArchName.split(":")[0], "T": T, "H": H,
                  "fwd_ms": round(ksum(pf), 4), "bwd_ms": round(ksum(pb), 4), "fwd_peak_mb": round(fwd_peak, 3), "bwd_peak_mb": round(bwd_peak, 3)}))
