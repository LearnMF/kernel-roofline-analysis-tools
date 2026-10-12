"""kra P-45b (on top of P-45): the fused tail's raw-gate / dt_bias tile loads move from the top of the
kt iteration to just before the end-of-kt barrier (issued after the scan, in flight across the
barrier wait).  Their live range then no longer spans the s4b body: P-45's TAIL instance hit 256 VGPRs
with 52 B (dense) / 88 B (varlen) of scratch.  Values unchanged vs P-45.
python3 p45b_s4bt_late_loads.py <tree with P-45>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wy.cuh"
t = P.read_text()
old = """        v4bh grw = v4bh{0, 0, 0, 0};                    // kra P-45: raw gate + dt_bias of the tile
        f4 dtb = f4{0.f, 0.f, 0.f, 0.f};
        if constexpr (TAIL) {
            grw = *reinterpret_cast<const v4bh*>(gr_ + rog + kc);
            dtb = ld16f(p.dt_bias + hh * kD + kc + 4 * q);
        }
"""
new = """        v4bh grw = v4bh{0, 0, 0, 0};                    // kra P-45: raw gate + dt_bias of the tile
        f4 dtb = f4{0.f, 0.f, 0.f, 0.f};                // (loaded late, before the end barrier: P-45b)
"""
assert t.count(old) == 1; t = t.replace(old, new)
old = """            if (m == 0) {                                    // block total -> table pad row 64 + A
                #pragma unroll
                for (int e = 0; e < 4; ++e) sm.TA[(4 * e + q) * TS + 64 + A] = sx[e];
            }
"""
new = old + """            grw = *reinterpret_cast<const v4bh*>(gr_ + rog + kc);   // kra P-45b: in flight across the barrier
            dtb = ld16f(p.dt_bias + hh * kD + kc + 4 * q);
"""
assert t.count(old) == 1; t = t.replace(old, new)
P.write_text(t)
print("P-45b APPLIED")
