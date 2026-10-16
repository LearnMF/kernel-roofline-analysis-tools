"""kra P-78a ABLATION (bitwise, timing only): what does s4b lose at 1 CTA/CU?  Design input for the s4b+tail segment
fusion (results/kda-segfusion-20261016): keeping a CTA's dq/dk/dg tiles (96 KB fp32) on chip needs ~96 more VGPRs
(s4b is at 236 VGPR / 26.9 KB LDS = 2 CTAs/CU), i.e. 1 CTA/CU.  KDA_G2_S4B_PADLDS=<bytes> launches s4b with that much
dynamic LDS (unused), so 26.9 KB + pad > 32 KB forces 1 CTA/CU with the code unchanged.
    python3 p78a_s4b_occ.py <tree>"""
import sys
from pathlib import Path

f = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wy.cuh"
s = f.read_text()
old = """        if (ctok) hipLaunchKernelGGL(p3_wy_s4b_kernel<true>, dim3(NT, H), dim3(256), 0, s, pb);
        else hipLaunchKernelGGL(p3_wy_s4b_kernel<false>, dim3(NT, H), dim3(256), 0, s, pb);"""
new = """        const int padl = (int)g2::g2_env_long("KDA_G2_S4B_PADLDS", 0);   // kra P-78a ablation
        if (ctok) hipLaunchKernelGGL(p3_wy_s4b_kernel<true>, dim3(NT, H), dim3(256), padl, s, pb);
        else hipLaunchKernelGGL(p3_wy_s4b_kernel<false>, dim3(NT, H), dim3(256), padl, s, pb);"""
assert s.count(old) == 1
f.write_text(s.replace(old, new))
print("P-78a APPLIED")
