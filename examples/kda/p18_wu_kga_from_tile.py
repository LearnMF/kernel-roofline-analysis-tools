"""kra P-18 (global route, round 3): wu's KgA leaves from the phase-3 tile as linear 16 B stores.
KgA (A form, L16-paired) carries the SAME values as Kn: kg = bf_rne(bf2f(k) * exp2f(gl - g)).
Phase 3 already builds kg in the tile (B-form native order) for the Kn emit; right after that
emit, and BEFORE the barrier that precedes the qg overwrite, the tile is gathered in A-form order
(8 scalar LDS reads per 16 B pair) -- no new phase, no new barrier (R18).  The scattered 8 B KgA
emit of phase 1 (ablation: ~0.38 ms of 4.59 at 8K/96) stays only on the keep-state path
(Kn == nullptr, no phase-3 kg tile).  Values unchanged -> bit-identical.
Usage: python3 p18_wu_kga_from_tile.py <tree root>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wu.cuh"
t = P.read_text()


def sub(old, new):
    global t
    assert t.count(old) == 1, old[:90]
    t = t.replace(old, new)


sub("template <bool VL>\n__global__ void __launch_bounds__(256) g2_wu_kernel(WuParams p) {",
    """// kra P-18: KgA (A form, L16 pairs) gathered from the B-form-ordered kg tile of phase 3.
// pair chunk c = native A positions o0..o0+3, o0+256..o0+259 (o0 = 512(c>>6) + 4(c&63));
// A position o <-> (j, v): v&3 = o&3, j&15 = (o>>2)&15, (v&15)>>2 = (o>>6)&3, v>>4 = (o>>8)&7,
// j>>4 = (o>>11)&3; B position of (j, v) = (j&3) + 4(v&15) + 64((j>>2)&3) + 256(j>>4) + 1024(v>>4).
__device__ __forceinline__ int wu_b_of_a(int o) {
    const int j = ((o >> 2) & 15) + 16 * ((o >> 11) & 3);
    const int v = (o & 3) + 4 * ((o >> 6) & 3) + 16 * ((o >> 8) & 7);
    return (j & 3) + 4 * (v & 15) + 64 * ((j >> 2) & 3) + 256 * (j >> 4) + 1024 * (v >> 4);
}
__device__ __forceinline__ void wu_emit16_a_from_b(short* dst, const short* tile, int tid) {
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int c = tid + 256 * k, o0 = 512 * (c >> 6) + 4 * (c & 63);
        v4bh lo, hi;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            lo[e] = tile[wu_tile_idx(wu_b_of_a(o0 + e))];
            hi[e] = tile[wu_tile_idx(wu_b_of_a(o0 + 256 + e))];
        }
        st16x2(dst + 8 * (size_t)c, lo, hi);
    }
}

template <bool VL>
__global__ void __launch_bounds__(256) g2_wu_kernel(WuParams p) {""")
# phase 1: the scattered KgA emit only on the keep-state path
sub("""            // KgA: A form, plain row nibble, P1/P2 block order
            *reinterpret_cast<v4bh*>(p.KgA + nat16(nblk + (size_t)(4 * (j & 15)""",
    """            // KgA: A form, plain row nibble, P1/P2 block order (kra P-18: default path emits it
            // from the phase-3 kg tile; only the keep-state path, without that tile, stores here)
            if (p.Kn == nullptr) *reinterpret_cast<v4bh*>(p.KgA + nat16(nblk + (size_t)(4 * (j & 15)""")
# phase 3: right after the Kn emit, gather KgA from the same tile
sub("""                __syncthreads();
                wu_emit16(p.Kn + nblk, tile, tid);
                __syncthreads();""",
    """                __syncthreads();
                wu_emit16(p.Kn + nblk, tile, tid);
                wu_emit16_a_from_b(p.KgA + nblk, tile, tid);   // kra P-18: same kg values, A form
                __syncthreads();""")
P.write_text(t)
print("P-18 APPLIED")
