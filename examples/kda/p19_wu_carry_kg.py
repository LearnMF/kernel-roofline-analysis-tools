"""kra P-19 (global route, round 3): wu's phase 3 stops re-reading k and the fp32 g.  Phase 1
already loads both with the SAME lane deal (t = tid + 256*it) and already computes kg (gkv) and
exp2f(g) (eg); they now stay in registers (16 + 32 VGPR; wu is LDS-bound at 2 CTAs/CU, so the
VGPR budget per wave is 256) and the default-path phase 3 only loads q:
    qg = bf_rne(bf2f(q) * eg)      kg -> tile = the carried gkv quad
-- character-identical expressions on identical inputs -> bit-identical.  48 KB less read per
CTA (~600 MB at 8K/96).  Keep-state path (Kn == nullptr) unchanged.
Usage: python3 p19_wu_carry_kg.py <tree root>      (on top of P-18)"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wu.cuh"
t = P.read_text()


def sub(old, new):
    global t
    assert t.count(old) == 1, old[:90]
    t = t.replace(old, new)


# phase 1: unrolled deal + carried values
sub("""        for (int t = tid; t < 2048; t += 256) {
            const int j = t >> 5, vq = (t & 31) << 2;
            const size_t so = sbase + (size_t)j * rstride + vq;
            const v4bh kv = g2_row_ok<M>(j, cl) ? *reinterpret_cast<const v4bh*>(p.k + so)
                                                 : v4bh{0, 0, 0, 0};""",
    """        v4bh kgr[8];                                     // kra P-19: kg quads carried to phase 3
        f4 egr[8];                                       // kra P-19: exp2f(g) carried to phase 3
        #pragma unroll
        for (int it = 0; it < 8; ++it) {
            const int t = tid + 256 * it;
            const int j = t >> 5, vq = (t & 31) << 2;
            const size_t so = sbase + (size_t)j * rstride + vq;
            const v4bh kv = g2_row_ok<M>(j, cl) ? *reinterpret_cast<const v4bh*>(p.k + so)
                                                 : v4bh{0, 0, 0, 0};""")
sub("""            short gv[4], gkv[4];
            #pragma unroll
            for (int dv = 0; dv < 4; ++dv) {
                gv[dv] = bf_rne(bf2f(kv[dv]) * eg[dv]);
                gkv[dv] = bf_rne(bf2f(kv[dv]) * exp2f(gl[dv] - gf[dv]));
            }""",
    """            short gv[4], gkv[4];
            #pragma unroll
            for (int dv = 0; dv < 4; ++dv) {
                gv[dv] = bf_rne(bf2f(kv[dv]) * eg[dv]);
                gkv[dv] = bf_rne(bf2f(kv[dv]) * exp2f(gl[dv] - gf[dv]));
            }
            kgr[it] = *reinterpret_cast<const v4bh*>(gkv);
            egr[it] = eg;""")
# phase 3 (default path): q only
sub("""                    const int tt = tid + 256 * it;
                    const int j = tt >> 5, vq = (tt & 31) << 2;
                    const f4 gf = qg_of(tt, qst[it]);
                    const v4bh kv = g2_row_ok<M>(j, cl)
                        ? *reinterpret_cast<const v4bh*>(p.k + sbase + (size_t)j * rstride + vq)
                        : v4bh{0, 0, 0, 0};
                    const f4 gl = *reinterpret_cast<const f4*>(sGl + vq);
                    #pragma unroll
                    for (int dv = 0; dv < 4; ++dv)
                        tile[wu_tile_idx(ibn_of(tt) + 4 * dv)] = bf_rne(bf2f(kv[dv]) * exp2f(gl[dv] - gf[dv]));""",
    """                    const int tt = tid + 256 * it;
                    const int j = tt >> 5, vq = (tt & 31) << 2;
                    // kra P-19: qg_of's value with the carried exp2f(g); kg = the carried quad
                    const v4bh qv = g2_row_ok<M>(j, cl)
                        ? *reinterpret_cast<const v4bh*>(p.q + sbase + (size_t)j * rstride + vq)
                        : v4bh{0, 0, 0, 0};
                    #pragma unroll
                    for (int dv = 0; dv < 4; ++dv) qst[it][dv] = bf_rne(bf2f(qv[dv]) * egr[it][dv]);
                    #pragma unroll
                    for (int dv = 0; dv < 4; ++dv)
                        tile[wu_tile_idx(ibn_of(tt) + 4 * dv)] = kgr[it][dv];""")
P.write_text(t)
print("P-19 APPLIED")
