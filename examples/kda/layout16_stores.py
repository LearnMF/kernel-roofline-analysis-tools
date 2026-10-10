"""kra L16 step 2: producers that hold BOTH halves of a pair in one lane write them as one
full 16 B store (validation P-8: 8 B stores into the paired layout touch twice the cache
lines per instruction; wu +80 us at 8K/H12).  Values/positions identical to step 1.
Usage: python3 layout16_stores.py <g2 kernel dir>"""
import sys
from pathlib import Path

K = Path(sys.argv[1])


def edit(path, pairs):
    t = path.read_text()
    for old, new, n in pairs:
        c = t.count(old)
        assert c == n, f"{path.name}: expected {n} x, found {c}: {old[:90]!r}"
        t = t.replace(old, new)
    path.write_text(t)


edit(K / "g2_common.h", [(
    "// fragments of sub-blocks 2j, 2j+1: p = base(512-aligned) + 512*j + 8*lane",
    """// store both halves of a pair at once (p = base(512-aligned) + 512*j + 8*lane)
__device__ __forceinline__ void st16x2(void* p, v4bh a, v4bh b) {
    *reinterpret_cast<v8bh_l16*>(p) = __builtin_shufflevector(a, b, 0, 1, 2, 3, 4, 5, 6, 7);
}
// fragments of sub-blocks 2j, 2j+1: p = base(512-aligned) + 512*j + 8*lane""", 1)])

edit(K / "g2_wu.cuh", [
    ("""            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) {
                short we[4];
                #pragma unroll
                for (int e = 0; e < 4; ++e) we[e] = bf_rne(acc[vt][e]);
                if (outC)   // C form is the C-fragment layout itself: one 8B vec store
                    *reinterpret_cast<v4bh*>(outC + nat16(nblk + (size_t)(vt + 8 * wv) * 256 + l * 4)) =
                        *reinterpret_cast<const v4bh*>(we);
                if (outA) { // A form, pi-staged: lane quad is native-contiguous
                    *reinterpret_cast<v4bh*>(outA + nat16(nblk + (size_t)(4 * (l & 15) + 64 * (l >> 4)
                        + 256 * vt + 2048 * wv))) = *reinterpret_cast<const v4bh*>(we);
                }
                if (outB) { // B form, pi-staged: t = 16wv+l&15, v = 16vt+4(l>>4)+e""",
     """            v4bh we8[8];
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt)
                #pragma unroll
                for (int e = 0; e < 4; ++e) we8[vt][e] = bf_rne(acc[vt][e]);
            // kra L16: C form (outC) and pi-staged A form (outA: 4(l&15) + 64(l>>4) == 4l) put
            // lane l's quad of sub-block vt + 8wv at slot l, so the vt pair (2j, 2j+1) is ONE
            // full 16 B store at nat16 position base(2j + 8wv) + 8l.
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const size_t pb = nblk + (size_t)(2 * j + 8 * wv) * 256 + 8 * l;
                if (outC) st16x2(outC + pb, we8[2 * j], we8[2 * j + 1]);
                if (outA) st16x2(outA + pb, we8[2 * j], we8[2 * j + 1]);
            }
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) {
                const v4bh we = we8[vt];
                if (outB) { // B form, pi-staged: t = 16wv+l&15, v = 16vt+4(l>>4)+e""", 1),
    ("""            for (int t = tid; t < 2048; t += 256)            // linear 8B global emit
                *reinterpret_cast<v4bh*>(p.Qn + nat16(nblk + 4 * t)) =
                    *reinterpret_cast<const v4bh*>(tile + wu_tile_idx(4 * t));""",
     """            for (int t = tid; t < 1024; t += 256) {          // kra L16: linear 16B pair emit
                const int o0 = 512 * (t >> 6) + 4 * (t & 63);   // sub-block 2g, lane t&63
                st16x2(p.Qn + nblk + 8 * (size_t)t,
                       *reinterpret_cast<const v4bh*>(tile + wu_tile_idx(o0)),
                       *reinterpret_cast<const v4bh*>(tile + wu_tile_idx(o0 + 256)));
            }""", 1),
])

edit(K / "g2_fwd_prep.cuh", [
    ("""        #pragma unroll
        for (int it = 0; it < 8; ++it) {
            const int U = threadIdx.x + 256 * it;
            const int dim = 16 * (U >> 8) + (U & 15), tok = 16 * ((U >> 6) & 3) + 4 * ((U >> 4) & 3);
            *reinterpret_cast<v4bh*>(p.Kn + nat16(nblk + 4 * (size_t)U)) =
                *reinterpret_cast<const v4bh*>(T2 + dim * kT2 + tok);
        }""",
     """        #pragma unroll
        for (int it = 0; it < 4; ++it) {                    // kra L16: 16B pair emit
            const int P = threadIdx.x + 256 * it;           // pair P = (group P>>6, lane P&63)
            v4bh h[2];
            #pragma unroll
            for (int s = 0; s < 2; ++s) {
                const int U = 128 * (P >> 6) + 64 * s + (P & 63);   // quad of sub-block 2g+s
                const int dim = 16 * (U >> 8) + (U & 15), tok = 16 * ((U >> 6) & 3) + 4 * ((U >> 4) & 3);
                h[s] = *reinterpret_cast<const v4bh*>(T2 + dim * kT2 + tok);
            }
            st16x2(p.Kn + nblk + 8 * (size_t)P, h[0], h[1]);
        }""", 1),
    ("""        #pragma unroll
        for (int vt = 0; vt < 8; ++vt)
            *reinterpret_cast<v4bh*>(out + nat16(nblk + (size_t)(vt + 8 * I) * 256 + 4 * l)) = pack4(acc[vt]);""",
     """        #pragma unroll
        for (int j = 0; j < 4; ++j)                         // kra L16: vt pair = one 16B store
            st16x2(out + nblk + (size_t)(2 * j + 8 * I) * 256 + 8 * l, pack4(acc[2 * j]), pack4(acc[2 * j + 1]));""", 1),
])
print("L16 STORES APPLIED")
