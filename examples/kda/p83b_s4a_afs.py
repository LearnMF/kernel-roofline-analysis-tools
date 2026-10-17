"""kra P-83b probe (bitwise, on top of P-83): with PF the AFS dA stage stages the A block from the registers P-83
already loaded (pav) instead of reloading it -- SQTT round 2 (map_waits -> g2_wy_hip.cuh:1066): 7.4% of s4a's
wave-cycles at 16K/12 wait on that reload, consumed right after issue.  Same quads, same mask, same LDS slots.
KDA_G2_S4A_AFSR=0 keeps the reload (probe switch, template int PF = 2 selects the reuse).
    python3 p83b_s4a_afs.py <dir holding g2_wy.cuh>"""
import sys
from pathlib import Path

f = Path(sys.argv[1]) / "g2_wy.cuh"
s = f.read_text()
old = """                *reinterpret_cast<v4bh*>(As2 + row * kPT + 4 * qc) = g2_row_ok<M>(row, cl)
                    ? *reinterpret_cast<const v4bh*>(A_ + row * RA + 4 * qc) : v4bh{0, 0, 0, 0};"""
new = """                *reinterpret_cast<v4bh*>(As2 + row * kPT + 4 * qc) = (PF == 2) ? pav[it_]   // kra P-83b
                    : (g2_row_ok<M>(row, cl) ? *reinterpret_cast<const v4bh*>(A_ + row * RA + 4 * qc) : v4bh{0, 0, 0, 0});"""
assert s.count(old) == 1
s = s.replace(old, new)
old2 = """            if (afs) { if (pf) hipLaunchKernelGGL((p3_wy_s4a_kernel<false, true, 1>), dim3(NT, H), dim3(256), 0, s, pa);"""
new2 = """            if (afs) { if (pf && g2::g2_env_long("KDA_G2_S4A_AFSR", 0)) hipLaunchKernelGGL((p3_wy_s4a_kernel<false, true, 2>), dim3(NT, H), dim3(256), 0, s, pa);   // kra P-83b
                       else if (pf) hipLaunchKernelGGL((p3_wy_s4a_kernel<false, true, 1>), dim3(NT, H), dim3(256), 0, s, pa);"""
assert s.count(old2) == 1
s = s.replace(old2, new2)
f.write_text(s)
print("P-83b APPLIED")
