"""Time g2_ops.wu alone (emit_p1 True/False) on the K3 bench inputs; per-process median.
PYTHONPATH=/opt/kda_env:<tree>:/opt/kda_env/takeover python3 wu_time.py T H"""
import sys
import torch
from hip_kda.ops import g2_ops

T, H = int(sys.argv[1]), int(sys.argv[2])
NT = T // 64
torch.manual_seed(0)
dev = "cuda"
mk = lambda s: (torch.randn(1, T, H, 128, device=dev) * s).to(torch.bfloat16)
q, k, v = mk(0.3), mk(0.3), mk(0.5)
step = -(torch.rand(1, T, H, 128, device=dev) * 0.05 + 0.001)
g = step.view(1, NT, 64, H, 128).cumsum(2).view(1, T, H, 128).contiguous()
beta = torch.rand(1, T, H, device=dev).float() * 0.8 + 0.1
A = (torch.eye(64, device=dev) + torch.tril(torch.randn(1, T, H, 64, 64, device=dev) * 0.05, -1)).to(torch.bfloat16)
A = A.reshape(1, T, H, 64).contiguous() if A.dim() == 5 and False else A[..., 0, :].contiguous() if False else \
    (torch.randn(1, T, H, 64, device=dev) * 0.05).to(torch.bfloat16)
for p1 in (True, False):
    for _ in range(3):
        g2_ops.wu(k, v, q, beta, A, g, H, NT, emit_p1=p1)
    torch.cuda.synchronize()
    ts = []
    for _ in range(20):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); g2_ops.wu(k, v, q, beta, A, g, H, NT, emit_p1=p1); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b) * 1000)
    ts.sort()
    print(f'{{"emit_p1": {str(p1).lower()}, "T": {T}, "H": {H}, "median_us": {ts[len(ts)//2]:.1f}}}')
