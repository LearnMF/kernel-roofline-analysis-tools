import sys
import torch
from hip_kda.fla import chunk_kda

T = int(sys.argv[1]) if len(sys.argv) > 1 else 64
H = int(sys.argv[2]) if len(sys.argv) > 2 else 1
print("shape", T, H, flush=True)
dev = "cuda"
q = torch.randn(1, T, H, 128, device=dev, dtype=torch.bfloat16).requires_grad_()
k = torch.randn(1, T, H, 128, device=dev, dtype=torch.bfloat16).requires_grad_()
v = torch.randn(1, T, H, 128, device=dev, dtype=torch.bfloat16).requires_grad_()
g = (torch.randn(1, T, H, 128, device=dev) * .3).bfloat16().requires_grad_()
beta = torch.randn(1, T, H, device=dev, dtype=torch.bfloat16).requires_grad_()
A_log = torch.randn(H, device=dev, dtype=torch.float32).requires_grad_()
dt_bias = torch.randn(H * 128, device=dev, dtype=torch.float32).requires_grad_()
cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
cu_cpu = cu.cpu()
print("fwd...", flush=True)
o = chunk_kda(q, k, v, g, beta, A_log=A_log, dt_bias=dt_bias, scale=None,
              output_final_state=False, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
              use_beta_sigmoid_in_kernel=False, safe_gate=True, lower_bound=-5.0,
              cu_seqlens=cu, cu_seqlens_cpu=cu_cpu)[0]
torch.cuda.synchronize()
print("fwd ok", o.shape, o.dtype, flush=True)
do = torch.randn_like(o)
o.backward(do)
torch.cuda.synchronize()
print("bwd ok", q.grad.abs().sum().item(), flush=True)
