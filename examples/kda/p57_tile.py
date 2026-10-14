"""kra P-57 probe (layout only, bitwise): dA (s4a -> s4b) and dAqk (dav -> s4b) stored per (chunk, head) as 16 blocks
of 16x16 fp32 (1 KB each, row-major inside), base ((i * H + hh) * 4096) floats, instead of row-major [T, H, 64].
Producers write whole 1 KB blocks per wave instruction; s4b's row reads become contiguous 1 KB, its transposed reads
4 x 64 B inside one block.  Dense path only (varlen keeps token-row addressing -> not served by this probe).
    python3 p57_tile.py <tree>"""
import sys
from pathlib import Path

g2 = Path(sys.argv[1]) / "csrc" / "g2"
REP = {
    "g2_wy.cuh": [
        ("*reinterpret_cast<f4*>(p.dA + cbAp + tr * RA + 16 * ut + 4 * q) = dak;",
         "*reinterpret_cast<f4*>(p.dA + ((size_t)hh * gridDim.x + i) * 4096 + (r * 4 + ut) * 256 + m * 16 + 4 * q) = dak;   // kra P-57"),
        ("    const float* aq_ = p.dAqk + cbA;\n    const float* dAg = p.dA + cbAp;",
         "    const float* aq_ = p.dAqk + ((size_t)hh * gridDim.x + i) * 4096;   // kra P-57: 16x16 fp32 blocks\n"
         "    const float* dAg = p.dA + ((size_t)hh * gridDim.x + i) * 4096;"),
        ("for (int ut = 0; ut < 4; ++ut) dak[ut] = ld16f(dAg + row * RA + 16 * ut + 4 * q);",
         "for (int ut = 0; ut < 4; ++ut) dak[ut] = ld16f(dAg + (A * 4 + ut) * 256 + m * 16 + 4 * q);"),
        ("aqA[b] = rok ? ld16f(aq_ + row * RA + 16 * b + 4 * q) : f4{0.f, 0.f, 0.f, 0.f};",
         "aqA[b] = rok ? ld16f(aq_ + (A * 4 + b) * 256 + m * 16 + 4 * q) : f4{0.f, 0.f, 0.f, 0.f};"),
        ("akB[b - A][x] = keep ? dAg[j * RA + row] : 0.f;   // dAT[row][j] == dA[j][row]",
         "akB[b - A][x] = keep ? dAg[((j >> 4) * 4 + A) * 256 + (j & 15) * 16 + m] : 0.f;   // dA[j][row], P-57 blocks"),
        ("aqB[b - A][x] = (keep && g2_row_ok<M>(j, cl)) ? aq_[j * RA + row] : 0.f;",
         "aqB[b - A][x] = (keep && g2_row_ok<M>(j, cl)) ? aq_[((j >> 4) * 4 + A) * 256 + (j & 15) * 16 + m] : 0.f;"),
    ],
    "g2_dav.cuh": [
        ("""                const int c = tid + 256 * it, t = c >> 4, j = (c & 15) * 4;
                if (!g2_row_ok<M>(t, cl)) continue;
                *reinterpret_cast<f4*>(p.dA + g2_rm_tok(t0 + t, hh, p.H, kBT) + j) =""",
         """                // kra P-57: one 16x16 fp32 block (1 KB) per wave instruction
                const int c = tid + 256 * it, bk = c >> 6, e = c & 63;
                const int t = (bk >> 2) * 16 + (e >> 2), j = (bk & 3) * 16 + (e & 3) * 4;
                if (!g2_row_ok<M>(t, cl)) continue;
                *reinterpret_cast<f4*>(p.dA + ((size_t)hh * gridDim.x + i) * 4096 + bk * 256 + e * 4) ="""),
    ],
}
for f, reps in REP.items():
    p = g2 / f
    s = p.read_text()
    for old, new in reps:
        assert s.count(old) == 1, (f, old[:70], s.count(old))
        s = s.replace(old, new)
    p.write_text(s)
print("P-57 APPLIED")
