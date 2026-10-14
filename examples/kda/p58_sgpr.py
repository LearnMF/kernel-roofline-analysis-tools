"""kra P-58 probe (codegen only, bitwise): the recurrence kernels' wave index is wave-uniform -> tell the compiler
(readfirstlane) so the per-step load / store addresses (base + (chunk + i, wave) block + lane offset) keep their
uniform part in SGPRs (global_* saddr form) instead of per-lane 64-bit VALU address arithmetic.
    python3 p58_sgpr.py <tree>"""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "csrc" / "g2" / "g2_fwd_v8.cuh"
s = p.read_text()
old = "const int tid = threadIdx.x, wv = tid >> 6, l = tid & 63;"
n = s.count(old)
assert n >= 1, n
s = s.replace(old, "const int tid = threadIdx.x, wv = __builtin_amdgcn_readfirstlane(tid >> 6), l = tid & 63;   // kra P-58")
p.write_text(s)
print("P-58 APPLIED", n)
