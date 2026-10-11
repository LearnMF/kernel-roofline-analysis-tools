"""kra P-21 (global route, round 4): dav's A^T fragments come from an LDS copy of the chunk's
64x64 Aqk block instead of 16 scalar 2 B global gathers per lane (ablation on the P-17/P-19
baseline: up to ~230 us of dav at 8K/96).  The slab is free between the dA emit and the doT
staging: the block is staged there with 2 x 16 B loads per thread (rows past the chunk end -> 0,
exactly the old g2_row_ok mask), every lane gathers its 16 values into registers, and a barrier
hands the slab back to the doT staging.  Causal mask (t >= tp) unchanged -> bit-identical.
Bank model: a gather instruction touches 4 rows x 16 columns = 32 distinct banks (conflict-free).
Usage: python3 p21_dav_at_lds.py <tree root>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_dav.cuh"
t = P.read_text()


def sub(old, new):
    global t
    assert t.count(old) == 1, old[:90]
    t = t.replace(old, new)


sub("""        // ---- pass 2: dv = A^T @ do, reduction over t (4 tiles), 8 v-tile outputs.
        // doT[v][t] staged transposed; A^T fragments are gathered straight from the
        // row-major Aqk (4 scalar loads per lane, L1-hot) with the causal zeroing --
        // FLA's upper-triangle garbage must not leak in.
        __syncthreads();""",
    """        // ---- pass 2: dv = A^T @ do, reduction over t (4 tiles), 8 v-tile outputs.
        // doT[v][t] staged transposed; A^T fragments are gathered straight from the
        // row-major Aqk (4 scalar loads per lane, L1-hot) with the causal zeroing --
        // FLA's upper-triangle garbage must not leak in.
        __syncthreads();
        // kra P-21: stage the 64x64 Aqk block in the (free) slab with 16 B loads and gather the
        // A^T fragments from LDS; the slab goes back to the doT staging after the barrier.
        v4bh afr[4];
        if (p.dVn != nullptr) {
            #pragma unroll
            for (int it = 0; it < 2; ++it) {                      // 64 rows x 8 chunks of 8
                const int c = tid + 256 * it, r = c >> 3, c8 = (c & 7) * 8;
                v4bh lo = v4bh{0, 0, 0, 0}, hi = v4bh{0, 0, 0, 0};
                if (g2_row_ok<M>(r, cl)) {
                    const short* src = p.A + g2_rm_tok(t0 + r, hh, p.H, kBT) + c8;
                    lo = *reinterpret_cast<const v4bh*>(src);
                    hi = *reinterpret_cast<const v4bh*>(src + 4);
                }
                *reinterpret_cast<v4bh*>(slab + (size_t)r * kPadJ3 + c8) = lo;
                *reinterpret_cast<v4bh*>(slab + (size_t)r * kPadJ3 + c8 + 4) = hi;
            }
            __syncthreads();
            const int tp = 16 * wv + (l & 15);
            #pragma unroll
            for (int js = 0; js < 4; ++js)
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int tt = 16 * js + 4 * (l >> 4) + e;
                    afr[js][e] = tt >= tp ? slab[(size_t)tt * kPadJ3 + tp] : (short)0;
                }
            __syncthreads();                                      // slab -> doT staging
        }""")
sub("""            for (int js = 0; js < 4; ++js) {                      // reduction over t
                v4bh af;
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int t = 16 * js + 4 * (l >> 4) + e;
                    af[e] = (t >= tp && g2_row_ok<M>(t, cl)) ?           // causal keep
                        *reinterpret_cast<const short*>(
                            p.A + g2_rm_tok(t0 + t, hh, p.H, kBT) + tp) : 0;
                }""",
    """            #pragma unroll
            for (int js = 0; js < 4; ++js) {                      // reduction over t
                const v4bh af = afr[js];                          // kra P-21: gathered from LDS""")
P.write_text(t)
print("P-21 APPLIED")
