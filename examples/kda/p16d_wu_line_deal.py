"""kra P-16d (global route, round 2): re-deal wu's phase-1/2 (row j, column quad vq) items so the
scattered emits write FULL 128 B lines (wu_deal_model.py: vB/kgB 16 -> 128 B per line touched,
KgA 32 -> 64), no new phases or barriers (R18).  P-16c showed the emit cost follows partially
written lines, not the instruction count.

Deal: wave w = 16-row group, lane l = 8a + b; iteration it = 4s + 2x + y:
    j  = 16w + (b & 3) + 8 (b >> 2) + 4y         (rows n, n+4: adjacent in the vB layout)
    vq = 4 (c & 3) + 32 (c >> 2) + 16x, c = a+8s (vq, vq+16: the L16 pair of KgA)
vB / kgB: rows (j, j+4) at one vq leave as one aligned 16 B store at y = 1; a line = 8 lanes b.
KgA: (vq, vq+16) at one row leave as one aligned 16 B nat16 pair at x = 1.
Every value (and the kbT staging) is the same expression on the same (j, vq) -> bit-identical.
Usage: python3 p16d_wu_line_deal.py <tree root>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wu.cuh"
t = P.read_text()


def sub(old, new):
    global t
    assert t.count(old) == 1, old[:90]
    t = t.replace(old, new)


sub("template <bool VL>\n__global__ void __launch_bounds__(256) g2_wu_kernel(WuParams p) {",
    """// kra P-16d: line-oriented deal of the 2048 (row, column-quad) items of phases 1/2.
__device__ __forceinline__ int wu_deal_j(int tid, int it) {
    const int b = tid & 7;
    return 16 * (tid >> 6) + (b & 3) + 8 * (b >> 2) + 4 * (it & 1);
}
__device__ __forceinline__ int wu_deal_vq(int tid, int it) {
    const int c = ((tid & 63) >> 3) + 8 * (it >> 2);
    return 4 * (c & 3) + 32 * (c >> 2) + 16 * ((it >> 1) & 1);
}

template <bool VL>
__global__ void __launch_bounds__(256) g2_wu_kernel(WuParams p) {""")

# ---- phase 1
sub("""        for (int t = tid; t < 2048; t += 256) {
            const int j = t >> 5, vq = (t & 31) << 2;
            const size_t so = sbase + (size_t)j * rstride + vq;
            const v4bh kv = g2_row_ok<M>(j, cl) ? *reinterpret_cast<const v4bh*>(p.k + so)
                                                 : v4bh{0, 0, 0, 0};""",
    """        v4bh kgB0, kgA0[2];                              // kra P-16d: first halves of the pairs
        #pragma unroll
        for (int it = 0; it < 8; ++it) {
            const int j = wu_deal_j(tid, it), vq = wu_deal_vq(tid, it);
            const size_t so = sbase + (size_t)j * rstride + vq;
            const v4bh kv = g2_row_ok<M>(j, cl) ? *reinterpret_cast<const v4bh*>(p.k + so)
                                                 : v4bh{0, 0, 0, 0};""")
sub("""            *reinterpret_cast<v4bh*>(p.kgB + wu_vb_quad(j, vq, nblk3)) =
                *reinterpret_cast<const v4bh*>(gv);
            // KgA: A form, plain row nibble, P1/P2 block order
            *reinterpret_cast<v4bh*>(p.KgA + nat16(nblk + (size_t)(4 * (j & 15)
                + 64 * ((vq & 15) >> 2) + 256 * (vq >> 4) + 2048 * (j >> 4)))) =
                *reinterpret_cast<const v4bh*>(gkv);""",
    """            // kgB: rows (j-4, j) at this vq -> one aligned 16 B store into the vB layout
            if (it & 1) st16x2(p.kgB + wu_vb_quad(j - 4, vq, nblk3), kgB0, *reinterpret_cast<const v4bh*>(gv));
            else kgB0 = *reinterpret_cast<const v4bh*>(gv);
            // KgA: A form, plain row nibble, P1/P2 block order; (vq-16, vq) = its L16 pair
            if ((it >> 1) & 1)
                st16x2(p.KgA + nat16(nblk + (size_t)(4 * (j & 15) + 64 * (((vq - 16) & 15) >> 2)
                           + 256 * ((vq - 16) >> 4) + 2048 * (j >> 4))),
                       kgA0[it & 1], *reinterpret_cast<const v4bh*>(gkv));
            else kgA0[it & 1] = *reinterpret_cast<const v4bh*>(gkv);""")

# ---- phase 2
sub("""        for (int t = tid; t < 2048; t += 256) {
            const int j = t >> 5, vq = (t & 31) << 2;
            const size_t so = sbase + (size_t)j * rstride + vq;
            const v4bh vv = g2_row_ok<M>(j, cl) ? *reinterpret_cast<const v4bh*>(p.v + so)
                                                 : v4bh{0, 0, 0, 0};""",
    """        v4bh vB0;                                        // kra P-16d: row j of the (j, j+4) pair
        #pragma unroll
        for (int it = 0; it < 8; ++it) {
            const int j = wu_deal_j(tid, it), vq = wu_deal_vq(tid, it);
            const size_t so = sbase + (size_t)j * rstride + vq;
            const v4bh vv = g2_row_ok<M>(j, cl) ? *reinterpret_cast<const v4bh*>(p.v + so)
                                                 : v4bh{0, 0, 0, 0};""")
sub("""            *reinterpret_cast<v4bh*>(p.vB + wu_vb_quad(j, vq, nblk3)) = vv;
        }
        if (p.stop == 4) return;""",
    """            if (it & 1) st16x2(p.vB + wu_vb_quad(j - 4, vq, nblk3), vB0, vv);
            else vB0 = vv;
        }
        if (p.stop == 4) return;""")
P.write_text(t)
print("P-16d APPLIED")
