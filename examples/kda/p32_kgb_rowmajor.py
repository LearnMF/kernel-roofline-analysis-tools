"""kra P-32 (round 10): kgB = bf16(k * 2^g) becomes ROW-MAJOR inside its (chunk, head) block.
wu writes it from the same (j, vq) quads it loads, so 32 lanes now store one full 256 B row (it was
an 8 B scatter into the B-fragment layout: ablation on the current baseline, wu 1.151x at 8K/96).
Its consumers gather the fragments themselves: s4a already stages the per-kt 2 KB kgB tile through
LDS (kra P-28), so only the staging thread's source address changes (fragment (st, lane) =
kg[16 st + pi16(lane & 15)][16 kt + 4 (lane >> 4) .. +3]); the fused S4 kernel and the legacy P3
kernel (non-default routes) read the same element through the same formula.  Values unchanged ->
bit-identical.        Usage: python3 p32_kgb_rowmajor.py <tree root>"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) / "csrc" / "g2"


def sub(f, old, new, count=1):
    p = ROOT / f
    t = p.read_text()
    assert t.count(old) == count, (f, old[:80], t.count(old))
    p.write_text(t.replace(old, new))


sub("g2_wu.cuh", "*reinterpret_cast<v4bh*>(p.kgB + wu_vb_quad(j, vq, nblk3)) =",
    "*reinterpret_cast<v4bh*>(p.kgB + nblk3 + (size_t)j * kK + vq) =   // kra P-32: row-major")
# s4a (split route, production): the P-28 staging source
sub("g2_wy.cuh", """            const bf16* kgsrc = kgB + (((tid >> 6) * 8) * 64 + (tid & 63)) * 4;
            *reinterpret_cast<v4bh*>(&KGs[0][0] + 4 * tid) = ld8(kgsrc);            // tile 0
            v4bh kgreg = ld8(kgsrc + 1 * 256);                                      // tile 1""",
    """            // kra P-32: kgB is row-major [64][kK]; fragment (st, lane) = row 16 st + pi16(lane & 15)
            const bf16* kgsrc = kgB + (16 * (tid >> 6) + pi16(tid & 15)) * kK + 4 * ((tid & 63) >> 4);
            *reinterpret_cast<v4bh*>(&KGs[0][0] + 4 * tid) = ld8(kgsrc);            // tile 0
            v4bh kgreg = ld8(kgsrc + 16);                                           // tile 1""")
sub("g2_wy.cuh", """                        kgreg = ld8(kgsrc + (kt + 2) * 256);""",
    """                        kgreg = ld8(kgsrc + (kt + 2) * 16);   // kra P-32""")
# legacy P3 (g2_wy) and the fused S4 kernel: same element, row-major address
# (no trailing comment: the legacy P3 site is a one-line loop body)
sub("g2_wy.cuh", "const v4bh kgb = ld8(kgB + ((st * 8 + kt) * 64 + l) * 4);",
    "const v4bh kgb = ld8(kgB + (16 * st + pi16(l & 15)) * kK + 16 * kt + 4 * (l >> 4));")
sub("g2_wy_s4_kernel.inc", "const v4bh kgb = ld8(kgB + ((st * 8 + kt) * 64 + l) * 4);",
    "const v4bh kgb = ld8(kgB + (16 * st + pi16(l & 15)) * kK + 16 * kt + 4 * (l >> 4));")
print("P-32 APPLIED")
