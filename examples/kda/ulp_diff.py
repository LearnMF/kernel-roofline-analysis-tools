"""Element-wise difference of two bitwise_gate dumps, in bf16 ULPs (for non-bitwise evaluation trees such as P-93).
    python3 ulp_diff.py BASE.pt CAND.pt"""
import sys

import torch


def ulps(a, b):
    # bf16 ULP distance via the monotone integer mapping of the bit patterns
    def key(x):
        i = x.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -(i & 0x7FFF), i)
    return (key(a) - key(b)).abs()


A, B = torch.load(sys.argv[1]), torch.load(sys.argv[2])
for n in A:
    a, b = A[n], B[n]
    if not torch.is_tensor(a):
        continue
    if a.dtype != torch.bfloat16:
        d = (a.double() - b.double())
        print(f"{n:9s} {str(a.dtype):15s} differing {int((a != b).sum()):>10d}/{a.numel():<10d} "
              f"rel_l2(diff) {float(d.norm() / a.double().norm().clamp_min(1e-30)):.3e}")
        continue
    u = ulps(a, b)
    tot = u.numel()
    h = [int((u == k).sum()) for k in (0, 1, 2)]
    print(f"{n:9s} bf16  identical {h[0] / tot:7.3%}  1 ulp {h[1] / tot:7.3%}  2 ulp {h[2] / tot:7.4%}  "
          f">2 ulp {(tot - sum(h)) / tot:7.4%}  max {int(u.max())} ulp  "
          f"rel_l2(diff) {float((a.double() - b.double()).norm() / a.double().norm().clamp_min(1e-30)):.3e}")
