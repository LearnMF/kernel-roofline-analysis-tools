"""kra L16 step 3 (wu): the two B-form outputs that were stride-4 SCALAR global stores (Kn, Wb)
go through the LDS tile (as Qn already did, T7) and leave as linear 16 B pair stores.  In the
paired layout each scalar store touched twice the cache lines (validation P-8: wu +55 us with
Kn, +20 us from Wb).  Kn moves from the phase-1 fill to phase 3 (the slab is busy in phase 1):
its value is recomputed there with the character-identical expression from the same k / g /
sGl inputs (k is L2-resident: the CTA read it in phase 1).  Values/positions identical.
Usage: python3 layout16_wu.py <g2 kernel dir>"""
import sys
from pathlib import Path

K = Path(sys.argv[1])
p = K / "g2_wu.cuh"
t = p.read_text()


def rep(old, new):
    global t
    assert t.count(old) == 1, (old[:80], t.count(old))
    t = t.replace(old, new)


# helper: CTA-wide linear 16 B pair emit of an 8192-element tile (B form, wu_tile_idx swizzle)
rep("""template <bool VL>
__global__ void __launch_bounds__(256) g2_wu_kernel(WuParams p) {""",
    """// kra L16: emit this CTA's 8192-element native block from the LDS tile as linear 16 B pairs
// (pair t = sub-blocks 2(t>>6), 2(t>>6)+1 at lane t&63 -> nat16 position 8t).
__device__ __forceinline__ void wu_emit16(short* dst, const short* tile, int tid) {
    for (int t = tid; t < 1024; t += 256) {
        const int o0 = 512 * (t >> 6) + 4 * (t & 63);
        st16x2(dst + 8 * (size_t)t, *reinterpret_cast<const v4bh*>(tile + wu_tile_idx(o0)),
               *reinterpret_cast<const v4bh*>(tile + wu_tile_idx(o0 + 256)));
    }
}

template <bool VL>
__global__ void __launch_bounds__(256) g2_wu_kernel(WuParams p) {""")

# Wb (outB): stage the scalars in the (now dead) slab, then one linear pair emit
rep("""            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) {
                const v4bh we = we8[vt];
                if (outB) { // B form, pi-staged: t = 16wv+l&15, v = 16vt+4(l>>4)+e
                    const int ib = ((l & 15) & 3) + 16 * (l >> 4) + 64 * ((l & 15) >> 2)
                                   + 256 * wv + 1024 * vt;
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) outB[nat16(nblk + ib + 4 * e)] = we[e];
                }
            }
        };""",
    """            if (outB) { // B form, pi-staged: t = 16wv+l&15, v = 16vt+4(l>>4)+e
                // kra L16: scalars -> LDS tile (the slab: every wave's MMAC reads are done
                // after the barrier) -> linear 16 B pair emit; the barrier after the emit
                // protects the tile from the next phase's slab fill.
                short* tile = slab;
                __syncthreads();
                #pragma unroll
                for (int vt = 0; vt < 8; ++vt) {
                    const int ib = ((l & 15) & 3) + 16 * (l >> 4) + 64 * ((l & 15) >> 2)
                                   + 256 * wv + 1024 * vt;
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) tile[wu_tile_idx(ib + 4 * e)] = we8[vt][e];
                }
                __syncthreads();
                wu_emit16(outB + nblk, tile, tid);
                __syncthreads();
            }
        };""")

# Kn: drop the phase-1 scalar scatter ...
rep("""            // Kn: B form -- the quad is 4 consecutive shorts at an e-dependent
            // (generally unaligned) offset, so scalar stride-1 stores
            const int ibn = (j & 3) + 4 * (vq & 15) + 64 * ((j >> 2) & 3)
                            + 256 * (j >> 4) + 1024 * (vq >> 4);
            if (p.Kn != nullptr) {                          // null: keep-state backward
                #pragma unroll
                for (int dv = 0; dv < 4; ++dv) p.Kn[nat16(nblk + ibn + 4 * dv)] = gkv[dv];
            }
        }""",
    """            // Kn (B form): emitted in phase 3 through the LDS tile (kra L16)
        }""")

# ... and emit it in phase 3 through the tile, before Qn
rep("""        {
            short* tile = slab;                             // 64x128 in native order
            for (int t = tid; t < 2048; t += 256) {
                const int j = t >> 5, vq = (t & 31) << 2;
                const size_t so = sbase + (size_t)j * rstride + vq;
                const v4bh qv""",
    """        {
            short* tile = slab;                             // 64x128 in native order
            if (p.Kn != nullptr) {                          // null: keep-state backward
                // kra L16: kg recomputed with phase 1's exact expression from the same
                // k / g / sGl inputs, B-form scalars -> tile -> linear 16 B pair emit.
                __syncthreads();                            // every wave's u-MMAC slab reads done
                for (int t = tid; t < 2048; t += 256) {
                    const int j = t >> 5, vq = (t & 31) << 2;
                    const size_t so = sbase + (size_t)j * rstride + vq;
                    const v4bh kv = g2_row_ok<M>(j, cl) ? *reinterpret_cast<const v4bh*>(p.k + so)
                                                         : v4bh{0, 0, 0, 0};
                    const f4 gf = ld16f(p.g + sbase + (size_t)g2_row_g<M>(j, cl) * rstride + vq);
                    const f4 gl = *reinterpret_cast<const f4*>(sGl + vq);
                    const int ibn = (j & 3) + 4 * (vq & 15) + 64 * ((j >> 2) & 3)
                                    + 256 * (j >> 4) + 1024 * (vq >> 4);
                    #pragma unroll
                    for (int dv = 0; dv < 4; ++dv)
                        tile[wu_tile_idx(ibn + 4 * dv)] = bf_rne(bf2f(kv[dv]) * exp2f(gl[dv] - gf[dv]));
                }
                __syncthreads();
                wu_emit16(p.Kn + nblk, tile, tid);
                __syncthreads();
            }
            for (int t = tid; t < 2048; t += 256) {
                const int j = t >> 5, vq = (t & 31) << 2;
                const size_t so = sbase + (size_t)j * rstride + vq;
                const v4bh qv""")
rep("""            for (int t = tid; t < 1024; t += 256) {          // kra L16: linear 16B pair emit
                const int o0 = 512 * (t >> 6) + 4 * (t & 63);   // sub-block 2g, lane t&63
                st16x2(p.Qn + nblk + 8 * (size_t)t,
                       *reinterpret_cast<const v4bh*>(tile + wu_tile_idx(o0)),
                       *reinterpret_cast<const v4bh*>(tile + wu_tile_idx(o0 + 256)));
            }""",
    """            wu_emit16(p.Qn + nblk, tile, tid);              // kra L16: linear 16B pair emit""")
p.write_text(t)
print("L16 WU APPLIED")
