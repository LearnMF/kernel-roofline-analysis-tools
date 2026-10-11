"""kra P-27b (round 9): P-27's k/q prefetch, issued AFTER the iteration's last consumed global load
(the dwT B-fragments of the epilogue product) and pinned there with sched_barrier.  P-27 issued it at
the top of the iteration, before this kt's gv/gn/pfh/pfd/dwT loads: vmcnt retires in issue order, so
every wait on those younger loads also drained the prefetch (s4b 0.977-0.998x).  Issued last, the
prefetch is older only than the NEXT iteration's loads, whose waits come ~one iteration later.
Values unchanged -> bit-identical.        Usage: python3 p27b_s4b_kq_prefetch_late.py <tree root>"""
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
    // kra P-27b: k/q fragments one kt ahead, issued after the iteration's last consumed load
    v4bh kkN = rok ? ld8(k_ + ro) : v4bh{0, 0, 0, 0};
    v4bh qqN = rok ? ld8(q_ + ro) : v4bh{0, 0, 0, 0};
    #pragma unroll 1
    for (int kt = 0; kt < 8; ++kt) {
        const int buf = kt & 1, kc = 16 * kt;
        const v4bh kk = kkN, qq = qqN;
        int4 pfh, pfd;""")
sub("""        const f4 gv = ld16f(g_ + rog + kc);
        const v4bh kk = rok ? ld8(k_ + ro + kc) : v4bh{0, 0, 0, 0};
        const v4bh qq = rok ? ld8(q_ + ro + kc) : v4bh{0, 0, 0, 0};
        const f4 gn = ld16f(g_ + rgn + kc);""",
    """        const f4 gv = ld16f(g_ + rog + kc);
        const f4 gn = ld16f(g_ + rgn + kc);""")
sub("""        for (int st = 0; st < 4; ++st)
            acc = mmac(aa[st], ld8(dwTg + (kc + pm) * kBT + 16 * st + 4 * q), acc);
""",
    """        for (int st = 0; st < 4; ++st)
            acc = mmac(aa[st], ld8(dwTg + (kc + pm) * kBT + 16 * st + 4 * q), acc);
        __builtin_amdgcn_sched_barrier(0);
        if (kt + 1 < 8) {                          // kra P-27b
            kkN = rok ? ld8(k_ + ro + kc + 16) : v4bh{0, 0, 0, 0};
            qqN = rok ? ld8(q_ + ro + kc + 16) : v4bh{0, 0, 0, 0};
        }
        __builtin_amdgcn_sched_barrier(0);
""")
P.write_text(t)
print("P-27b APPLIED")
