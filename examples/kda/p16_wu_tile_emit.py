"""kra P-16 (global route, round 2): wu's kgB and vB leave through the LDS tile as linear 16 B
stores.  The staged ablation showed the two scattered 8 B emits into the (pi16-permuted, P3
order) vB layout cost ~1.26 ms of wu's 4.55 ms at 8K/96 (rule R10: store cost ~ address span).
Default path (Kn != null): kgB = bf_rne(bf2f(k) * exp2f(g)) is computed in phase 3 (which already
loads k and g with the same masks) -- character-identical to phase 1's expression; vB = raw v is
re-read in phase 3 (L2-resident); both are written to the tile as 8 B quads at their vB-layout
positions and emitted linearly.  The keep-state path (Kn == null) keeps the original emits.
Usage: python3 p16_wu_tile_emit.py <tree root>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wu.cuh"
t = P.read_text()


def sub(old, new):
    global t
    assert t.count(old) == 1, old[:90]
    t = t.replace(old, new)


# helper: plain linear emit of an 8192-element block from the swizzled tile
sub("template <bool VL>\n__global__ void __launch_bounds__(256) g2_wu_kernel(WuParams p) {",
    """// kra P-16: tile swizzle for the vB layout.  A wave's quad writes sit 16 quads apart (the
// layout's minor index is the pi16 row), i.e. on the SAME LDS bank pair; XOR-ing the quad's low
// 4 bits with its next 4 bits spreads them over all 16 bank pairs (and keeps the emit's
// consecutive-quad reads spread too).  Any bijection works: writer and reader share it.
__device__ __forceinline__ int wu_vbt_idx(int x) {
    const int q = x >> 2;
    return ((q ^ ((q >> 4) & 15)) << 2) | (x & 3);
}
// plain (unpaired) linear 16 B emit of a native 8192-element block (vB / kgB are not L16-paired)
__device__ __forceinline__ void wu_emit16_plain(short* dst, const short* tile, int tid) {
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int x = 8 * (tid + 256 * k);
        st16x2(dst + x, *reinterpret_cast<const v4bh*>(tile + wu_vbt_idx(x)),
               *reinterpret_cast<const v4bh*>(tile + wu_vbt_idx(x + 4)));
    }
}
// position of vB-layout quad (row j, columns vq..vq+3) inside its 8192-element block
__device__ __forceinline__ int wu_vb_pos(int j, int vq) {
    return 4 * pi16(j & 15) + 64 * ((vq & 15) >> 2) + 256 * (vq >> 4) + 2048 * (j >> 4);
}

template <bool VL>
__global__ void __launch_bounds__(256) g2_wu_kernel(WuParams p) {""")
# phase 1: kgB scattered emit only on the keep-state path
sub("""            *reinterpret_cast<v4bh*>(p.kgB + wu_vb_quad(j, vq, nblk3)) =
                *reinterpret_cast<const v4bh*>(gv);""",
    """            if (p.Kn == nullptr)    // kra P-16: default path emits kgB from phase 3 via the tile
                *reinterpret_cast<v4bh*>(p.kgB + wu_vb_quad(j, vq, nblk3)) =
                    *reinterpret_cast<const v4bh*>(gv);""")
# phase 2: vB scattered emit only on the keep-state path
sub("""            *reinterpret_cast<v4bh*>(p.vB + wu_vb_quad(j, vq, nblk3)) = vv;
        }
        if (p.stop == 4) return;""",
    """            if (p.Kn == nullptr)    // kra P-16: default path emits vB from phase 3 via the tile
                *reinterpret_cast<v4bh*>(p.vB + wu_vb_quad(j, vq, nblk3)) = vv;
        }
        if (p.stop == 4) return;""")
# phase 3: keep kgB values in registers next to qg
sub("""            short* tile = slab;                             // 64x128 in native order""",
    """            short* tile = slab;                             // 64x128 in native order
            v4bh kgst[8];                                   // kra P-16: kgB quads (default path)""")
sub("""                    const f4 gl = *reinterpret_cast<const f4*>(sGl + vq);
                    #pragma unroll
                    for (int dv = 0; dv < 4; ++dv)
                        tile[wu_tile_idx(ibn_of(tt) + 4 * dv)] = bf_rne(bf2f(kv[dv]) * exp2f(gl[dv] - gf[dv]));
                }""",
    """                    const f4 gl = *reinterpret_cast<const f4*>(sGl + vq);
                    #pragma unroll
                    for (int dv = 0; dv < 4; ++dv)
                        tile[wu_tile_idx(ibn_of(tt) + 4 * dv)] = bf_rne(bf2f(kv[dv]) * exp2f(gl[dv] - gf[dv]));
                    #pragma unroll
                    for (int dv = 0; dv < 4; ++dv)      // phase 1's kgB expression, same inputs
                        kgst[it][dv] = bf_rne(bf2f(kv[dv]) * exp2f(gf[dv]));
                }""")
# after the Qn emit: kgB and vB tile rounds (default path only)
sub("""            __syncthreads();
            wu_emit16(p.Qn + nblk, tile, tid);              // kra L16: linear 16B pair emit""",
    """            __syncthreads();
            wu_emit16(p.Qn + nblk, tile, tid);              // kra L16: linear 16B pair emit
            if (p.Kn != nullptr) {                          // kra P-16: kgB, then vB, via the tile
                __syncthreads();                            // Qn emit reads done
                #pragma unroll
                for (int it = 0; it < 8; ++it) {
                    const int tt = tid + 256 * it, j = tt >> 5, vq = (tt & 31) << 2;
                    *reinterpret_cast<v4bh*>(tile + wu_vbt_idx(wu_vb_pos(j, vq))) = kgst[it];
                }
                __syncthreads();
                wu_emit16_plain(p.kgB + nblk3, tile, tid);
                __syncthreads();
                #pragma unroll
                for (int it = 0; it < 8; ++it) {
                    const int tt = tid + 256 * it, j = tt >> 5, vq = (tt & 31) << 2;
                    *reinterpret_cast<v4bh*>(tile + wu_vbt_idx(wu_vb_pos(j, vq))) = g2_row_ok<M>(j, cl)
                        ? *reinterpret_cast<const v4bh*>(p.v + sbase + (size_t)j * rstride + vq)
                        : v4bh{0, 0, 0, 0};
                }
                __syncthreads();
                wu_emit16_plain(p.vB + nblk3, tile, tid);
            }""")
P.write_text(t)
print("P-16 APPLIED")
