"""kra P-90 (bitwise): prep_a loads the 4 row betas of the QgA/Kn pass before the loop.

SQTT round 3 at 16K/48 (map_waits -> g2_fwd_prep_hip.cuh:327): four vmcnt(0) waits, ~8.4% of prep_a's wave-cycles,
one per rb iteration -- each iteration loads its row's beta and, in the same iteration, stores Kb to global, so the
next beta load cannot move above that store (R59).  The betas are now loaded first (same addresses, same mask).
    python3 p90_prep_beta.py <dir holding g2_fwd_prep.cuh>"""
import sys
from pathlib import Path

f = Path(sys.argv[1]) / "g2_fwd_prep.cuh"
s = f.read_text()
old = """        ldref(7, gn);
        #pragma unroll
        for (int rb = 0; rb < 4; ++rb) {
            v4bh qg[2], kb[2];
            const int row = 16 * rb + m;
            const bool rok = g2_row_ok<M>(row, cl);
            const float bta = rok ? p.beta[g2_rm_tok(tok0 + row, hh, p.H, 1)] : 0.f;"""
new = """        ldref(7, gn);
        float btav[4];   // kra P-90: the 4 row betas in flight before the Kb stores (R59)
        #pragma unroll
        for (int rb = 0; rb < 4; ++rb)
            btav[rb] = g2_row_ok<M>(16 * rb + m, cl) ? p.beta[g2_rm_tok(tok0 + 16 * rb + m, hh, p.H, 1)] : 0.f;
        #pragma unroll
        for (int rb = 0; rb < 4; ++rb) {
            v4bh qg[2], kb[2];
            const int row = 16 * rb + m;
            const bool rok = g2_row_ok<M>(row, cl);
            const float bta = btav[rb];"""
assert s.count(old) == 1, s.count(old)
f.write_text(s.replace(old, new))
print("P-90 APPLIED")
