"""kra P-88 probe (bitwise): s4a / s4b on a head-fastest grid (KDA_G2_S4_HF=1 launches dim3(H, NT) instead of
dim3(NT, H); the kernels read (chunk, head) from whichever grid order was launched).

Three-platform study 2026-10-17 (results/kda-xplat-3way-20261017): per 12 heads, s4a / s4b grow with H -- 16K: s4a
479 / 571 / 609 / 868 us and s4b 1306 / 1359 / 1259 / 1699 us at H = 12 / 24 / 48 / 96 -- while tail / wu / prep / dav
stay flat.  At TP1 that is +5.6 ms (16K) / +9.6 ms (32K) over linear.  Hypothesis: with the chunk-fastest grid the
~160 co-resident CTAs are consecutive chunks of ONE head, spanning 160 x 64 tokens x (row stride H*256 B) -- the
page / TLB footprint of every row-major tensor doubles with H.  Head-fastest residents share 2-3 chunks across all
heads instead.  Each (chunk, head) CTA is independent (fixed-order reductions over db_part): values unchanged.
    python3 p88_s4_hf.py <dir holding g2_wy.cuh>"""
import sys
from pathlib import Path

f = Path(sys.argv[1]) / "g2_wy.cuh"
s = f.read_text()
LINE = "    const int i = blockIdx.x, hh = blockIdx.y, H = p.H;"
NEW = ("    // kra P-88: head-fastest grid when launched as dim3(H, NT) (H == NT: both orders are the same bijection)\n"
       "    const bool hf_ = gridDim.x == (unsigned)p.H && gridDim.y != (unsigned)p.H;\n"
       "    const int i = hf_ ? (int)blockIdx.y : (int)blockIdx.x, hh = hf_ ? (int)blockIdx.x : (int)blockIdx.y, H = p.H;")
a = s.index("p3_wy_s4a_kernel(WyS4AParams p_) {")
b = s.index('extern "C" int g2_wy_s4_split_r(')
mid = s[a:b]
assert mid.count(LINE) == 3, mid.count(LINE)
mid = mid.replace(LINE, NEW)
s = s[:a] + mid + s[b:]
a = s.index('extern "C" int g2_wy_s4_split_r(')
b = s.index('extern "C" int g2_wy_s4_split(', a)
ln = s[a:b]
n = ln.count("dim3(NT, H)")
assert n == 11, n
ln = ln.replace("dim3(NT, H)", "s4grd")
ln = ln.replace("        int afs_maxh = 12;", "        const dim3 s4grd = g2::g2_env_long(\"KDA_G2_S4_HF\", 0) ? dim3(H, NT) : dim3(NT, H);   // kra P-88\n"
                "        int afs_maxh = 12;", 1)
assert "const dim3 s4grd" in ln
# the B launch sits in a second block scope: give it its own copy
ln = ln.replace("        if (ctok) hipLaunchKernelGGL(p3_wy_s4b_kernel<true>, s4grd,",
                "        const dim3 s4grd = g2::g2_env_long(\"KDA_G2_S4_HF\", 0) ? dim3(H, NT) : dim3(NT, H);   // kra P-88\n"
                "        if (ctok) hipLaunchKernelGGL(p3_wy_s4b_kernel<true>, s4grd,", 1)
s = s[:a] + ln + s[b:]
f.write_text(s)
print("P-88 APPLIED")
