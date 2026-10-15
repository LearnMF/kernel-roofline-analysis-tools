"""kra P-66 probe (on top of P-63d): the bN selector MMAC starts from C = +0 (zero accumulator, no -0.0
materialization: P-63d's -0.0 C costs 4 v_mov per MMAC, ~16 per step on the recompute chain).
Exactness: C' = (+0) + x*1 + (+-0 terms) = x for x != 0; only x = -0 would flip sign.  h is never -0: h0 = +0,
h <- h*d + upd with upd an MMAC sum started from +0 (never -0), and a + b = -0 in RNE needs both -0.  The bitwise
gate (incl. gate-extreme, where d underflows to 0) checks it.
    python3 p66_bn_zero_c.py <tree>"""
import sys
from pathlib import Path

f = Path(sys.argv[1]) / "csrc" / "g2" / "g2_fwd_v8.cuh"
s = f.read_text()
old = """                    const f4 nz = f4{-0.f, -0.f, -0.f, -0.f};
                    const f4 c = mmac(bsel, hx[vt], nz);"""
assert s.count(old) == 1
s = s.replace(old, """                    const f4 c = mmac(bsel, hx[vt], f4{0.f, 0.f, 0.f, 0.f});   // kra P-66: +0 C (h is never -0)""")
f.write_text(s)
print("P-66 APPLIED")
