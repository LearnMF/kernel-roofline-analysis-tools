"""kra P-13b: wu's rstd for the chunk's 64 rows staged ONCE per CTA in LDS (P-13 read one
scalar rstd per quad from global -- 24 dependent loads per thread, wu +11%).
Usage: python3 p13b_wu_rstd_lds.py <g2 kernel dir>"""
import re
import sys
from pathlib import Path

W = Path(sys.argv[1]) / "g2_wu.cuh"
t = W.read_text()
# the three P-13 sites -> LDS
n0 = t.count("p.k_rstd[so >> 7]") + t.count("p.q_rstd[so >> 7]") + t.count("p.k_rstd[(sbase + (size_t)j * rstride + vq) >> 7]")
assert n0 == 3, n0
t = t.replace("p.k_rstd[so >> 7]", "sRk[j]").replace("p.q_rstd[so >> 7]", "sRq[j]")
t = t.replace("p.k_rstd[(sbase + (size_t)j * rstride + vq) >> 7]", "sRk[j]")
# stage both rstd vectors right after sGl's preload (its barrier publishes them too)
m = re.search(r"\n( *)(ld16f\(p\.g \+ sbase \+ g2_row_g<M>\(63, cl\) \* rstride \+ 4 \* tid\);)", t)
assert m, "sGl preload not found"
ind = " " * 8          # statement level of the preload block
stage = (f"\n{ind}if (p.k_rstd != nullptr && tid < 128) {{                // kra P-13b: rstd rows -> LDS once\n"
         f"{ind}    const int jr = tid & 63;\n"
         f"{ind}    const float* rs = tid < 64 ? p.k_rstd : p.q_rstd;\n"
         f"{ind}    (tid < 64 ? sRk : sRq)[jr] = g2_row_ok<M>(jr, cl) ? rs[(sbase >> 7) + (size_t)jr * p.H] : 1.f;\n"
         f"{ind}}}")
t = t[:m.end()] + stage + t[m.end():]
# declare the LDS arrays next to sGl
m2 = re.search(r"__shared__[^\n;]*\bsGl\b[^\n;]*;", t)
assert m2, "sGl declaration not found"
t = t[:m2.end()] + "\n    __shared__ float sRk[64], sRq[64];   // kra P-13b" + t[m2.end():]
W.write_text(t)
print("P-13b APPLIED")
