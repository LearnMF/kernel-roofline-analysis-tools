"""Full (mangled-demangled) names + per-iteration time of every kernel in one fwd+bwd (flash-train, dense, K3 route)."""
import sys, collections, torch, torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
import flash_train.torch as ft
T, H = map(int, sys.argv[1].split("x"))
g = torch.Generator(device="cuda").manual_seed(1)
r = lambda *s: torch.randn(*s, device="cuda", generator=g)
X = {"q": F.normalize(r(1, T, H, 128), dim=-1).bfloat16(), "k": F.normalize(r(1, T, H, 128), dim=-1).bfloat16(),
     "v": (r(1, T, H, 128) * .3).bfloat16(), "g": (r(1, T, H, 128) * .3).bfloat16(), "beta": r(1, T, H).sigmoid(),
     "A_log": torch.log(1 + 15 * torch.rand(H, device="cuda", generator=g)), "dt_bias": r(H * 128) * .1}
L = {k: v.requires_grad_(True) for k, v in X.items()}
D = r(1, T, H, 128).bfloat16(); CU = torch.tensor([0, T], dtype=torch.int32, device="cuda")
KW = dict(A_log=L["A_log"], dt_bias=L["dt_bias"], use_qk_l2norm_in_kernel=False, use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0, cu_seqlens=CU)
def step():
    for t in L.values(): t.grad = None
    ft.chunk_kda(L["q"], L["k"], L["v"], L["g"], L["beta"], **KW)[0].backward(D)
for _ in range(2): step()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as pr:
    for _ in range(3): step()
    torch.cuda.synchronize()
c = collections.Counter()
for e in pr.events():
    if e.device_type.name == "CUDA": c[e.name] += e.device_time / 3
for n, t in c.most_common(20): print(f"{t:9.1f}  {n[:150]}")
