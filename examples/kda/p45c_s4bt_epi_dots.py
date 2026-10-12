"""kra P-45c (on top of P-45b): the l2norm row dots are computed in the epilogue from the re-read
dq/dk and qn/kn (which the epilogue loads anyway) instead of two in-loop accumulators, and the gate's
A = exp(A_log[h]) is made wave-uniform (readfirstlane -> SGPR).  Frees 3 VGPRs of the TAIL instance's
loop (P-45b: 256 VGPRs + 28/48 B scratch).  Same association as P-45 per lane (columns in kt order).
python3 p45c_s4bt_epi_dots.py <tree with P-45b>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wy.cuh"
t = P.read_text()


def rep(old, new):
    global t
    assert t.count(old) == 1, (old[:80], t.count(old))
    t = t.replace(old, new)


rep("""    float pdq = 0.f, pdk = 0.f, dAs = 0.f, Ag = 0.f;          // kra P-45 (TAIL)
    const short* gr_ = nullptr;
    if constexpr (TAIL) { Ag = expf(p.A_log[hh]); gr_ = p.g_raw + cb; }""",
    """    float dAs = 0.f, Ag = 0.f;                                 // kra P-45 (TAIL)
    const short* gr_ = nullptr;
    if constexpr (TAIL) {
        Ag = __int_as_float(__builtin_amdgcn_readfirstlane(__float_as_int(expf(p.A_log[hh]))));
        gr_ = p.g_raw + cb;
    }""")
rep("""                pdq += oq[e] * bf2f(qq[e]);                  // l2norm row dots (qq/kk = qn/kn tile)
                pdk += ok[e] * bf2f(kk[e]);
""", "")
rep("""        const float dqd = xsum16(pdq), dkd = xsum16(pdk);
        if (rok) {""", """        float pdq = 0.f, pdk = 0.f;                          // kra P-45c: dots from the re-read values
        if (rok) {""")
rep("""                kv[kt] = ld8(k_ + ro + 16 * kt);
            }
            #pragma unroll
            for (int kt = 0; kt < 8; ++kt) {
                v4bh oq_, ok_;""", """                kv[kt] = ld8(k_ + ro + 16 * kt);
            }
            #pragma unroll
            for (int kt = 0; kt < 8; ++kt)
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    pdq += dqv[kt][e] * bf2f(qv[kt][e]);
                    pdk += dkv[kt][e] * bf2f(kv[kt][e]);
                }
            const float dqd = xsum16(pdq), dkd = xsum16(pdk);
            #pragma unroll
            for (int kt = 0; kt < 8; ++kt) {
                v4bh oq_, ok_;""")
P.write_text(t)
print("P-45c APPLIED")
