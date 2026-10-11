"""kra P-27 (round 9): s4b_tail's row-major k/q fragments (2 x 8 B per lane per kt, 16-row x 32 B
segments) are issued at the top of each kt and consumed one 8-MMAC pass later, so their global
latency sits on the step's critical path (ablation: s4b -8.5% at 8K/12, -6.2% at 8K/96).  They are
now prefetched one kt ahead (the textbook's gfx936 rule: issue the next beat's loads before this
beat's math).  Values unchanged -> bit-identical.      Usage: python3 p27_s4b_kq_prefetch.py <tree>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wy.cuh"
t = P.read_text()


def sub(old, new):
    global t
    assert t.count(old) == 1, (old[:80], t.count(old))
    t = t.replace(old, new)


sub("""    float dbacc = 0.f;
    #pragma unroll 1
    for (int kt = 0; kt < 8; ++kt) {
        const int buf = kt & 1, kc = 16 * kt;
        int4 pfh, pfd;""",
    """    float dbacc = 0.f;
    // kra P-27: k/q fragments one kt ahead
    v4bh kkN = rok ? ld8(k_ + ro) : v4bh{0, 0, 0, 0};
    v4bh qqN = rok ? ld8(q_ + ro) : v4bh{0, 0, 0, 0};
    #pragma unroll 1
    for (int kt = 0; kt < 8; ++kt) {
        const int buf = kt & 1, kc = 16 * kt;
        const v4bh kk = kkN, qq = qqN;
        if (kt + 1 < 8) {
            kkN = rok ? ld8(k_ + ro + kc + 16) : v4bh{0, 0, 0, 0};
            qqN = rok ? ld8(q_ + ro + kc + 16) : v4bh{0, 0, 0, 0};
        }
        int4 pfh, pfd;""")
sub("""        const f4 gv = ld16f(g_ + rog + kc);
        const v4bh kk = rok ? ld8(k_ + ro + kc) : v4bh{0, 0, 0, 0};
        const v4bh qq = rok ? ld8(q_ + ro + kc) : v4bh{0, 0, 0, 0};
        const f4 gn = ld16f(g_ + rgn + kc);""",
    """        const f4 gv = ld16f(g_ + rog + kc);
        const f4 gn = ld16f(g_ + rgn + kc);""")
P.write_text(t)
print("P-27 APPLIED")
