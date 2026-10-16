"""kra P-78b ABLATION (bitwise, timing only): fwd_prep at lower occupancy -- design input for merging prep_a + prep_b
(R/Kb scratch ~265 MB at 16K/12): R (fp32, 20 KB per chunk*head) kept in LDS next to prep_a's 21 KB arena gives
~41.5 KB per CTA = 1 CTA/CU (prep_a runs 3 CTAs/CU today).  KDA_G2_PREP_PADLDS=<bytes> launches BOTH prep kernels with
that much unused dynamic LDS.
    python3 p78b_prep_occ.py <tree>"""
import sys
from pathlib import Path

f = Path(sys.argv[1]) / "csrc" / "g2" / "g2_fwd_prep.cuh"
s = f.read_text()
s0 = s
s = s.replace("""    const dim3 grid(NT, H), blk(256);
    if (gate) {""", """    const dim3 grid(NT, H), blk(256);
    const int padl = (int)g2::g2_env_long("KDA_G2_PREP_PADLDS", 0);   // kra P-78b ablation
    if (gate) {""")
for kern in ("(g2::r4::g2_fwd_prep_a_kernel<1, true>)", "(g2::r4::g2_fwd_prep_a_kernel<1, false>)",
             "(g2::r4::g2_fwd_prep_a_kernel<0, true>)", "(g2::r4::g2_fwd_prep_a_kernel<0, false>)"):
    o = f"hipLaunchKernelGGL({kern}, grid, blk, 0, s, pa)"
    assert s.count(o) == 1, o
    s = s.replace(o, f"hipLaunchKernelGGL({kern}, grid, blk, padl, s, pa)")
for kern in ("g2::r4::g2_fwd_prep_b_kernel<true>", "g2::r4::g2_fwd_prep_b_kernel<false>"):
    o = f"hipLaunchKernelGGL({kern}, grid, blk, 0, s, pb)"
    assert s.count(o) == 1, o
    s = s.replace(o, f"hipLaunchKernelGGL({kern}, grid, blk, padl, s, pb)")
assert s != s0
f.write_text(s)
print("P-78b APPLIED")
