"""One FLA chunk_kda forward+backward (the K3 call of kda_op_bench_nv.py, same inputs), bracketed by
cudaProfilerStart/Stop after warm-up, for Nsight Compute:
    ncu --profile-from-start off ... python3 nv_fla_once.py T H [fla|fla_nr]"""
import sys

import torch

T, H = int(sys.argv[1]), int(sys.argv[2])
arm = sys.argv[3] if len(sys.argv) > 3 else "fla"
from fla.ops.kda import chunk_kda  # noqa: E402

gen = torch.Generator(device="cuda").manual_seed(T + H)
r = lambda *s: torch.randn(*s, device="cuda", generator=gen)
X = {"q": (r(1, T, H, 128) * .3).bfloat16(), "k": (r(1, T, H, 128) * .3).bfloat16(),
     "v": (r(1, T, H, 128) * .3).bfloat16(), "g": (r(1, T, H, 128) * .3).bfloat16(),
     "beta": r(1, T, H).sigmoid(), "A_log": torch.log(1 + 15 * torch.rand(H, device="cuda", generator=gen)),
     "dt_bias": r(H * 128) * .1}
D_O = r(1, T, H, 128).bfloat16()
CU = torch.tensor([0, T], dtype=torch.int32, device="cuda")
L = {k: v.requires_grad_(True) for k, v in X.items()}
kw = dict(A_log=L["A_log"], dt_bias=L["dt_bias"], scale=None, output_final_state=False,
          use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=False,
          safe_gate=True, lower_bound=-5.0, cu_seqlens=CU, cu_seqlens_cpu=CU.cpu(),
          disable_recompute=(arm == "fla_nr"))


def step():
    o = chunk_kda(L["q"], L["k"], L["v"], L["g"], L["beta"], **kw)[0]
    torch.autograd.grad(o, list(L.values()), D_O)


for _ in range(3):
    step()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
step()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print("NV_FLA_ONCE_DONE")
