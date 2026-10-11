"""kra P-31 (round 10): the varlen instance of p3_wy_s4a (p3_wy_s4a_kernel<true>, the flash-train /
packed-document path) issues ALL its global memory ops as flat_* (187 flat, 0 global; every other
kernel's VL instance uses global_*).  VL=true instantiates the g2_vl_dispatch body twice (full and
masked chunk), the compiler outlines the big lambda, and the kernel-argument pointers it captures by
reference lose their global address space.  Flat ops also count on lgkmcnt, so LDS waits drain
global loads too.  Forcing the lambda inline restores global_* (values unchanged -> bit-identical).
Usage: python3 p31_s4a_vl_inline.py <tree root>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wy.cuh"
t = P.read_text()
i = t.index("p3_wy_s4a_kernel(WyS4AParams p) {")
old = "    g2_vl_dispatch<VL>(cl, [&](auto mode_) {"
j = t.index(old, i)
assert j - i < 2000, "dispatch not in s4a"
t = t[:j] + "    g2_vl_dispatch<VL>(cl, [&](auto mode_) __attribute__((always_inline)) {   // kra P-31" + t[j + len(old):]
P.write_text(t)
print("P-31 APPLIED")
