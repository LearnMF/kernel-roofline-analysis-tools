"""kra P-16c (global route, round 2): wu's scattered vB / kgB emits become half as many, twice as
wide -- WITHOUT new phases or barriers (P-16's tile rounds serialized and lost, rule R18).

In the vB layout the quad of row j sits at 4*pi16(j&15) + ...; pi16(n+4) = pi16(n) + 1 for
n < 12, so rows j and j+4 are ADJACENT quads, and for n in {0..3, 8..11} the pair is 16 B
aligned.  Phases 1 and 2 iterate (row, column quad) as t = tid + 256*it -> j = t >> 5; here the
same 2048 (j, vq) items are re-dealt so that a lane's iterations 2g, 2g+1 hold rows
j0 = 16g + n(r) and j0 + 4 at the SAME vq (r = tid >> 5, n(r) = (r & 3) + 8*((r >> 2) & 1)):
the pair leaves as one st16x2 to the vB / kgB layout.  Every value and every other store is the
same expression on the same (j, vq) as before -> bit-identical.
Usage: python3 p16c_wu_pair_rows.py <tree root>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wu.cuh"
t = P.read_text()


def sub(old, new):
    global t
    assert t.count(old) == 1, old[:90]
    t = t.replace(old, new)


sub("template <bool VL>\n__global__ void __launch_bounds__(256) g2_wu_kernel(WuParams p) {",
    """// kra P-16c: deal the 2048 (row, column-quad) items of phases 1/2 so that iterations 2g, 2g+1
// of a lane hold rows j and j+4 (adjacent, 16 B aligned quads in the vB layout) at one vq.
__device__ __forceinline__ int wu_pair_row(int tid, int it) {
    const int r = tid >> 5;
    return 16 * (it >> 1) + (r & 3) + 8 * ((r >> 2) & 1) + 4 * (it & 1);
}

template <bool VL>
__global__ void __launch_bounds__(256) g2_wu_kernel(WuParams p) {""")

# ---- phase 1
sub("""        for (int t = tid; t < 2048; t += 256) {
            const int j = t >> 5, vq = (t & 31) << 2;
            const size_t so = sbase + (size_t)j * rstride + vq;
            const v4bh kv = g2_row_ok<M>(j, cl) ? *reinterpret_cast<const v4bh*>(p.k + so)
                                                 : v4bh{0, 0, 0, 0};""",
    """        v4bh kgB0;                                       // kra P-16c: row j of the (j, j+4) pair
        #pragma unroll
        for (int it = 0; it < 8; ++it) {
            const int j = wu_pair_row(tid, it), vq = (tid & 31) << 2;
            const size_t so = sbase + (size_t)j * rstride + vq;
            const v4bh kv = g2_row_ok<M>(j, cl) ? *reinterpret_cast<const v4bh*>(p.k + so)
                                                 : v4bh{0, 0, 0, 0};""")
sub("""            *reinterpret_cast<v4bh*>(p.kgB + wu_vb_quad(j, vq, nblk3)) =
                *reinterpret_cast<const v4bh*>(gv);""",
    """            if (it & 1)                                  // rows j-4, j: one aligned 16 B store
                st16x2(p.kgB + wu_vb_quad(j - 4, vq, nblk3), kgB0, *reinterpret_cast<const v4bh*>(gv));
            else
                kgB0 = *reinterpret_cast<const v4bh*>(gv);""")

# ---- phase 2
sub("""        for (int t = tid; t < 2048; t += 256) {
            const int j = t >> 5, vq = (t & 31) << 2;
            const size_t so = sbase + (size_t)j * rstride + vq;
            const v4bh vv = g2_row_ok<M>(j, cl) ? *reinterpret_cast<const v4bh*>(p.v + so)
                                                 : v4bh{0, 0, 0, 0};""",
    """        v4bh vB0;                                        // kra P-16c: row j of the (j, j+4) pair
        #pragma unroll
        for (int it = 0; it < 8; ++it) {
            const int j = wu_pair_row(tid, it), vq = (tid & 31) << 2;
            const size_t so = sbase + (size_t)j * rstride + vq;
            const v4bh vv = g2_row_ok<M>(j, cl) ? *reinterpret_cast<const v4bh*>(p.v + so)
                                                 : v4bh{0, 0, 0, 0};""")
sub("""            *reinterpret_cast<v4bh*>(p.vB + wu_vb_quad(j, vq, nblk3)) = vv;
        }
        if (p.stop == 4) return;""",
    """            if (it & 1)                                  // rows j-4, j: one aligned 16 B store
                st16x2(p.vB + wu_vb_quad(j - 4, vq, nblk3), vB0, vv);
            else
                vB0 = vv;
        }
        if (p.stop == 4) return;""")
P.write_text(t)
print("P-16c APPLIED")
