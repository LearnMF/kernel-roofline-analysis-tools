"""kra L16 step 3b (wu): when Kn is emitted, phase 3 reads k, q and g ONCE: kg goes to the LDS
tile and qg is held in registers (8 quads / thread) until the Kn emit frees the tile.  Values
are the character-identical expressions.  Usage: python3 layout16_wu2.py <g2 kernel dir>"""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "g2_wu.cuh"
t = p.read_text()
k0 = t.index("            if (p.Kn != nullptr) {                          // null: keep-state backward\n                // kra L16: kg recomputed")
k1 = t.index("            wu_emit16(p.Qn + nblk, tile, tid);              // kra L16: linear 16B pair emit")
old = t[k0:k1]
assert "for (int t = tid; t < 2048; t += 256) {\n                const int j = t >> 5, vq = (t & 31) << 2;\n                const size_t so = sbase + (size_t)j * rstride + vq;\n                const v4bh qv" in old
new = """            auto qg_of = [&](int t, short* qgv) {           // T7 value, unchanged
                const int j = t >> 5, vq = (t & 31) << 2;
                const size_t so = sbase + (size_t)j * rstride + vq;
                const v4bh qv = g2_row_ok<M>(j, cl) ? *reinterpret_cast<const v4bh*>(p.q + so)
                                                     : v4bh{0, 0, 0, 0};
                const f4 gf = ld16f(p.g + sbase + (size_t)g2_row_g<M>(j, cl) * rstride + vq);
                #pragma unroll
                for (int dv = 0; dv < 4; ++dv) qgv[dv] = bf_rne(bf2f(qv[dv]) * exp2f(gf[dv]));
                return gf;
            };
            auto ibn_of = [](int t) {
                const int j = t >> 5, vq = (t & 31) << 2;
                return (j & 3) + 4 * (vq & 15) + 64 * ((j >> 2) & 3) + 256 * (j >> 4) + 1024 * (vq >> 4);
            };
            if (p.Kn != nullptr) {                          // null: keep-state backward
                // kra L16: one pass over k / q / g -- kg (phase 1's exact expression, same
                // inputs) -> tile -> linear 16 B pair emit to Kn; qg waits in registers.
                short qst[8][4];
                __syncthreads();                            // every wave's u-MMAC slab reads done
                #pragma unroll
                for (int it = 0; it < 8; ++it) {
                    const int tt = tid + 256 * it;
                    const int j = tt >> 5, vq = (tt & 31) << 2;
                    const f4 gf = qg_of(tt, qst[it]);
                    const v4bh kv = g2_row_ok<M>(j, cl)
                        ? *reinterpret_cast<const v4bh*>(p.k + sbase + (size_t)j * rstride + vq)
                        : v4bh{0, 0, 0, 0};
                    const f4 gl = *reinterpret_cast<const f4*>(sGl + vq);
                    #pragma unroll
                    for (int dv = 0; dv < 4; ++dv)
                        tile[wu_tile_idx(ibn_of(tt) + 4 * dv)] = bf_rne(bf2f(kv[dv]) * exp2f(gl[dv] - gf[dv]));
                }
                __syncthreads();
                wu_emit16(p.Kn + nblk, tile, tid);
                __syncthreads();
                #pragma unroll
                for (int it = 0; it < 8; ++it)
                    #pragma unroll
                    for (int dv = 0; dv < 4; ++dv) tile[wu_tile_idx(ibn_of(tid + 256 * it) + 4 * dv)] = qst[it][dv];
            } else {
                for (int t = tid; t < 2048; t += 256) {
                    short qgv[4];
                    qg_of(t, qgv);
                    #pragma unroll
                    for (int dv = 0; dv < 4; ++dv) tile[wu_tile_idx(ibn_of(t) + 4 * dv)] = qgv[dv];
                }
            }
            __syncthreads();
"""
t = t[:k0] + new + t[k1:]
p.write_text(t)
print("L16 WU2 APPLIED")
