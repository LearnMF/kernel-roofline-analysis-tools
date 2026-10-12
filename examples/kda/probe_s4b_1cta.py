"""Probe for F2 feasibility: run p3_wy_s4b at ONE CTA per CU (dynamic-LDS top-up to 33 KB, the
P-24 mechanism) to measure s4b's sensitivity to the occupancy a register-resident dq/dk epilogue
would force.  Values unchanged.      python3 probe_s4b_1cta.py <tree root>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wy.cuh"
t = P.read_text()
old = """        if (ctok) hipLaunchKernelGGL(p3_wy_s4b_kernel<true>, dim3(NT, H), dim3(256), 0, s, pb);
        else hipLaunchKernelGGL(p3_wy_s4b_kernel<false>, dim3(NT, H), dim3(256), 0, s, pb);"""
new = """        // PROBE: one CTA per CU (static 32 KB + top-up -> 33 KB, floor(64/33) = 1)
        if (ctok) hipLaunchKernelGGL(p3_wy_s4b_kernel<true>, dim3(NT, H), dim3(256),
                                     g2_lds_topup(p3_wy_s4b_kernel<true>, 33 * 1024), s, pb);
        else hipLaunchKernelGGL(p3_wy_s4b_kernel<false>, dim3(NT, H), dim3(256),
                                g2_lds_topup(p3_wy_s4b_kernel<false>, 33 * 1024), s, pb);"""
assert t.count(old) == 1
P.write_text(t.replace(old, new))
print("PROBE s4b 1 CTA/CU applied")
