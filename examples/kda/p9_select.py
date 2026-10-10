"""kra P-9 selection by measurement: keep the compile-time v8o4 instances only where they won
(mode 2 forward keep-state, mode 3 backward recompute: ~5% faster); mode 1 (forward, o only)
measured 4% slower and goes back to the generic instance.  Usage: python3 p9_select.py <g2 dir>"""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "g2_fwd_v8.cuh"
s = p.read_text()
old = """    int mode = 0;
    if (O == nullptr && Ob != nullptr && !Vn && !HbN && !ANv) mode = 1;
    else if"""
new = """    // Mode 1 (forward, o only) is NOT selected: measured 4% slower (kra P-9) -- without the
    // conservative waits the compiler clusters the loads and VMEM issue stalls return, while the
    // generic instance's waits happen to pace them.  Modes 2/3 measured ~5% faster.
    int mode = 0;
    if"""
assert s.count(old) == 1
s = s.replace(old, new)
c1 = "            case 1: hipLaunchKernelGGL((g2_fwd_h_v8o4_kernel<VL, 1>), grid, dim3(256), 0, s, p); break;\n"
assert s.count(c1) == 1
s = s.replace(c1, "")
p.write_text(s)
print("P-9 SELECT APPLIED")
