"""kra P-40 (A-class, production s4b at 2 CTAs/CU): the P-37h3 change in s4b_tail -- the LDS table
writes that every lane could do with the SAME value at the SAME address no longer sit behind
exec-mask branches (cs2: msum leaves the sum in all 16 lanes of the row; cs1: xsum16 leaves it in
the 4 lanes of the column; gM: the m==8 lane's gv equals the ep/en shuffle partner every lane
already fetched), and the rlast dg correction is a select (never "+ 0").  Values unchanged.
python3 p40_s4b_prod_nobranch.py <tree>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wy.cuh"
t = P.read_text()
head = "__device__ __forceinline__ void s4b_tail(const WyS4BParams& p, S4BSmem& sm,"
i = t.index(head)
j = t.index("\n}\n", i)
seg = t[i:j]


def rep(old, new):
    global seg
    assert seg.count(old) == 1, (old[:80], seg.count(old))
    seg = seg.replace(old, new)


rep("""                const float c = msum(kd[e]);
                if (m == 0) sm.cs2[buf][A][4 * q + e] = c;""",
    """                const float c = msum(kd[e]);
                sm.cs2[buf][A][4 * q + e] = c;            // kra P-40: all 16 lanes hold c""")
rep("""            hd = xsum16(hd);
            if (q == 0) sm.cs1[buf][A][pm] = hd;""",
    """            hd = xsum16(hd);
            sm.cs1[buf][A][pm] = hd;                      // kra P-40: 4 lanes, same value""")
rep("""        f4 ep, en;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float d = gv[e] - __shfl(gv[e], 16 * q + 8, 64);""",
    """        f4 ep, en, g8;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            g8[e] = __shfl(gv[e], 16 * q + 8, 64);        // kra P-40: row 16A+8's g (== m==8 lane's gv)
            const float d = gv[e] - g8[e];""")
rep("""        if (m == 8) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) sm.gM[A][4 * e + q] = gv[e];
        }""",
    """        #pragma unroll
        for (int e = 0; e < 4; ++e) sm.gM[A][4 * e + q] = g8[e];   // kra P-40: every lane, same value""")
rep("""            if (row == rlast) {
                const int c = 4 * q + e;
                const float s1 = ((sm.cs1[buf][0][c] + sm.cs1[buf][1][c]) + sm.cs1[buf][2][c]) + sm.cs1[buf][3][c];
                const float s2 = ((sm.cs2[buf][0][c] + sm.cs2[buf][1][c]) + sm.cs2[buf][2][c]) + sm.cs2[buf][3][c];
                dg += s1 * ex2(gn[e]) + s2;
            }""",
    """            {                                             // kra P-40: select, never "+ 0"
                const int c = 4 * q + e;
                const float s1 = ((sm.cs1[buf][0][c] + sm.cs1[buf][1][c]) + sm.cs1[buf][2][c]) + sm.cs1[buf][3][c];
                const float s2 = ((sm.cs2[buf][0][c] + sm.cs2[buf][1][c]) + sm.cs2[buf][2][c]) + sm.cs2[buf][3][c];
                const float dgc = dg + (s1 * ex2(gn[e]) + s2);
                dg = row == rlast ? dgc : dg;
            }""")
t = t[:i] + seg + t[j:]
P.write_text(t)
print("P-40 APPLIED")
