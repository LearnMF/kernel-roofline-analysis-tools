"""kra P-10 selection by measurement: the compile-time production instance helps the 8-wave dhu
(v8b: 1.027x at 8K/H12) but hurts the 4-wave dhu (v8b4: 0.916x at 8K/H48, loads cluster as in
P-9 mode 1) -> only v8b uses it.  Usage: python3 p10_select.py <g2 kernel dir>"""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "g2_engine.cuh"
s = p.read_text()
old = """    if (dhu4) {
        if (prod) hipLaunchKernelGGL(g2_dhu_v8b4_kernel<1>, dim3(kK / kBV, nseg * H), dim3(256), 0, s, p);
        else      hipLaunchKernelGGL(g2_dhu_v8b4_kernel<0>, dim3(kK / kBV, nseg * H), dim3(256), 0, s, p);
    } else {"""
new = """    if (dhu4) {
        // v8b4: the compile-time instance measured 8% SLOWER (kra P-10) -> generic only
        hipLaunchKernelGGL(g2_dhu_v8b4_kernel<0>, dim3(kK / kBV, nseg * H), dim3(256), 0, s, p);
    } else {"""
assert s.count(old) == 1
s = s.replace(old, new)
p.write_text(s)
print("P-10 SELECT APPLIED")
