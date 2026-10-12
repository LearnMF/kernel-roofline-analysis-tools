"""kra P-37d (round 12, s4b rewrite step 2b, A-class): double-buffered tables in the pipelined s4b
-> ONE barrier per kt instead of two.

TA/TQ/TK and gM gain a [kt & 1] buffer index (LDS 51,200 -> 64,512 B, still < 64 KiB).  Iteration
kt now reads stage kt, writes tables(kt) into buffer kt&1, waits for stage kt+1 (vmcnt(3): only the
three stores of kt-1 are newer) and passes ONE barrier that publishes both; then stage kt+2 is
issued into the consumed stage and the back half runs.  Hazards: tables/gM/cs of buffer b are
rewritten two iterations later, after the next barrier, so every reader has finished; stage reads
all precede the barrier.  Values unchanged (only addresses).
Apply after p37_s4b_pipe.py (+ p37c).        python3 p37d_s4b_dbl_tables.py <tree>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wy.cuh"
t = P.read_text()
a = t.index("// ================================================================ kra P-37: pipelined s4b")
b = t.index('extern "C" int g2_wy_s4_split_r2(', a) if 'g2_wy_s4_split_r2(' in t[a:] else t.index('extern "C" int g2_wy_s4_split_r(', a)
seg = t[a:b]


def rep(old, new, count=1):
    global seg
    assert seg.count(old) == count, (old[:70], seg.count(old))
    seg = seg.replace(old, new)


rep("""struct S4BPSmem {
    float TA[16 * ipi::TS], TQ[16 * ipi::TS], TK[16 * ipi::TS];
    float gM[4][16];""", """struct S4BPSmem {
    float TA[2][16 * ipi::TS], TQ[2][16 * ipi::TS], TK[2][16 * ipi::TS];   // kra P-37d: [kt & 1]
    float gM[2][4][16];""")
rep("            sm.TA[o] = kf * en[e];\n            sm.TQ[o] = qf * ep[e];\n            sm.TK[o] = kf * beta_t * ep[e];",
    "            sm.TA[buf][o] = kf * en[e];\n            sm.TQ[buf][o] = qf * ep[e];\n            sm.TK[buf][o] = kf * beta_t * ep[e];")
rep("            for (int e = 0; e < 4; ++e) sm.gM[A][4 * e + q] = gv[e];",
    "            for (int e = 0; e < 4; ++e) sm.gM[buf][A][4 * e + q] = gv[e];")
rep("        const float gm_n = sm.gM[A][m];", "        const float gm_n = sm.gM[buf][A][m];")
rep("            f4 t = *reinterpret_cast<const f4*>(&sm.TA[m * TS + 16 * b + 4 * q]);\n            if (b < A) t *= ex2_diff(gm_n, sm.gM[b][m]);",
    "            f4 t = *reinterpret_cast<const f4*>(&sm.TA[buf][m * TS + 16 * b + 4 * q]);\n            if (b < A) t *= ex2_diff(gm_n, sm.gM[buf][b][m]);")
rep("            f4 tq = *reinterpret_cast<const f4*>(&sm.TQ[m * TS + 16 * b + 4 * q]);\n            f4 tk = *reinterpret_cast<const f4*>(&sm.TK[m * TS + 16 * b + 4 * q]);",
    "            f4 tq = *reinterpret_cast<const f4*>(&sm.TQ[buf][m * TS + 16 * b + 4 * q]);\n            f4 tk = *reinterpret_cast<const f4*>(&sm.TK[buf][m * TS + 16 * b + 4 * q]);")
rep("                const float s = ex2_diff(sm.gM[b][m], gm_n);", "                const float s = ex2_diff(sm.gM[buf][b][m], gm_n);")
rep("""        lds_barrier();                                     // tables visible; stage kt consumed
        if (kt + 2 < 8) s4bp_issue<A>(io, S, kt + 2);      // tile kt+2 -> the stage kt freed""",
    """        // kra P-37d: the ONE barrier of kt publishes tables(kt) AND stage kt+1 (issued one
        // iteration ago; only the three stores of kt-1 were issued after it)
        if (kt > 0 && kt + 1 < 8) s4bp_vmcnt<3>();
        lds_barrier();                                     // tables(kt) + stage kt+1 visible
        if (kt + 2 < 8) s4bp_issue<A>(io, S, kt + 2);      // tile kt+2 -> the stage kt freed""")
rep("""        // stage kt+1 must be complete before the barrier publishes it.  Ops this wave issued after
        // stage kt+1's group: [stores kt-1 (3)] + [stage kt+2 (NA) if issued] + [stores kt (3)].
        if (kt == 0) s4bp_vmcnt<3 + NA>();
        else if (kt + 2 < 8) s4bp_vmcnt<6 + NA>();
        else if (kt + 1 < 8) s4bp_vmcnt<6>();
        lds_barrier();                                     // table WAR + stage kt+1 visible
""", "")
t = t[:a] + seg + t[b:]
P.write_text(t)
print("P-37d (double-buffered tables, one barrier per kt) APPLIED")
