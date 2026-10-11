"""kra P-29 (round 9): s4b_tail's epilogue product reads its dwT B-fragments (4 x 8 B per lane per kt)
straight from global -- the same 2 KB tile for all 4 waves.  Now the CTA loads the tile ONCE per kt
(one coalesced 8 B load per thread, issued first in the iteration so its vmcnt wait does not drain
younger loads), stores it XOR-swizzled (column group cg ^ row) into a 2 KB LDS tile (conflict-free
fragment reads without padding: LDS 30,720 -> 32,768 B, still 2 CTAs/CU), and every wave reads its
fragments from LDS after the existing mid-iteration barrier.  The previous tile's reads end before
the end-of-iteration barrier, so the overwrite is race-free.  Values and MMAC order unchanged ->
bit-identical.  Rationale (P-27/P-27b): s4b is VMEM-issue bound; only fewer VMEM instructions help.
Usage: python3 p29_s4b_dwt_dedupe.py <tree root>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wy.cuh"
t = P.read_text()


def sub(old, new):
    global t
    assert t.count(old) == 1, (old[:80], t.count(old))
    t = t.replace(old, new)


sub("""    bf16 stg[2][2][8 * 256];
};""", """    bf16 stg[2][2][8 * 256];
    bf16 dwTs[16 * 64];          // kra P-29: dwT tile of step kt, column groups XOR-swizzled by row
};""")
sub("""        const int buf = kt & 1, kc = 16 * kt;
        int4 pfh, pfd;""",
    """        const int buf = kt & 1, kc = 16 * kt;
        // kra P-29: this kt's dwT tile, one 8 B load per thread (issued first: oldest in vmcnt)
        const int dwr = threadIdx.x >> 4, dwc = threadIdx.x & 15;
        const v4bh dwv = ld8(dwTg + (kc + dwr) * kBT + 4 * dwc);
        int4 pfh, pfd;""")
sub("""        if (m == 8) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) sm.gM[A][4 * e + q] = gv[e];
        }
        lds_barrier();""",
    """        if (m == 8) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) sm.gM[A][4 * e + q] = gv[e];
        }
        *reinterpret_cast<v4bh*>(&sm.dwTs[dwr * 64 + 4 * (dwc ^ dwr)]) = dwv;   // kra P-29
        lds_barrier();""")
sub("""            acc = mmac(aa[st], ld8(dwTg + (kc + pm) * kBT + 16 * st + 4 * q), acc);""",
    """            acc = mmac(aa[st], ld8(&sm.dwTs[pm * 64 + 4 * ((4 * st + q) ^ pm)]), acc);   // kra P-29""")
P.write_text(t)
print("P-29 APPLIED")
