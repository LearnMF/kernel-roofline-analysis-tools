"""kra P-15 (global route, round 2): wu without the sA LDS cache -- the two MMAC phases read
their A fragments (4 x 8 B per lane per phase) straight from the L2-resident row-major A, with
the same row mask as the old sA fill (VL rows r >= clen -> 0).  LDS 26,880 -> 18,176 B, so the
CU holds 3 CTAs (12 waves) instead of 2: wu is latency/occupancy limited (HBM 63%, VMEM 38%,
pipe 47% at 8K/48).  Values unchanged.     Usage: python3 p15_wu_no_sA.py <tree root>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wu.cuh"
t = P.read_text()


def sub(old, new):
    global t
    assert t.count(old) == 1, old[:90]
    t = t.replace(old, new)


sub("    __shared__ __align__(16) short sA[kBT * kPadA];\n", "")
sub("""        for (int t = tid; t < 1024; t += 256) {          // sA: 64x64 bf16 row-major
            const int r = t >> 4, c4 = (t & 15) << 2;
            *reinterpret_cast<v4bh*>(sA + r * kPadA + c4) = g2_row_ok<M>(r, cl)
                ? *reinterpret_cast<const v4bh*>(
                      p.A + g2_rm_tok(g2_row_tok<VL>(p.ctok, i) + r, hh, p.H, kBT) + c4)
                : v4bh{0, 0, 0, 0};
        }
        __syncthreads();   // sBeta/sGl/sA must land before the phase-1 fill reads them""",
    """        __syncthreads();   // sBeta/sGl must land before the phase-1 fill reads them""")
sub("""            for (int js = 0; js < 4; ++js) {
                const v4bh af = *reinterpret_cast<const v4bh*>(
                    sA + (size_t)(16 * wv + (l & 15)) * kPadA + 16 * js + (l >> 4) * 4);""",
    """            // kra P-15: A fragments straight from the L2-resident row-major A (was: the sA
            // LDS copy, which capped the CU at 2 CTAs); same row mask as the old sA fill.
            const int ar = 16 * wv + (l & 15);
            const bool aok = g2_row_ok<M>(ar, cl);
            const short* arow = p.A
                + g2_rm_tok(g2_row_tok<VL>(p.ctok, i) + ar, hh, p.H, kBT) + (l >> 4) * 4;
            for (int js = 0; js < 4; ++js) {
                const v4bh af = aok ? *reinterpret_cast<const v4bh*>(arow + 16 * js)
                                    : v4bh{0, 0, 0, 0};""")
P.write_text(t)
print("P-15 APPLIED")
