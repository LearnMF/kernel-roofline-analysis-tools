"""kra P-61 probe (bitwise): write_h stores h with one ds_write_b64 per (v-tile, k-block) instead of four ds_write_b16.

Critical path of the forward recurrence after P-60 (cp_step.py): write_h's 16 ds_write_b16 (+ their bf16 VALU) run
back-to-back right before barrier A, ~300-400 cycles of ~2860 per step.  GEMM2's C fragment gives lane (q4, m)
v = m and k = B-lane(q4 + 4e), i.e. four k's 4 apart.  prep now emits Kn with the 16 lanes of every sub-block
pi16-permuted (lane n carries K column pi16(n); store-side index change only), so the lane's four elements are the
consecutive k = 4 q4 + e and hB[v][k] takes one 8-byte write.  The per-k decays must follow the element order:
prep writes Gn in natural k order (was pi16-permuted to match the old C order).  t-contraction order unchanged
-> bit-identical.  Forward modes (0/1/2) of v8o4 only; modes 3/4 read wu's Kn and the backward Gn and keep the
old path; probe scope: forward does not take v8o4w.
    python3 p61_kn_perm.py <tree>"""
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "csrc" / "g2"


def sub(s, old, new, cnt=1):
    assert s.count(old) == cnt, (old[:80], s.count(old))
    return s.replace(old, new)


p = root / "g2_fwd_prep.cuh"
s = p.read_text()
s = sub(s, "                    gnp[(col >> 4) * 16 + (col & 3) * 4 + ((col & 15) >> 2)] = exp2f(refv[e]);   // kra P-52",
        "                    gnp[col] = exp2f(refv[e]);   // kra P-52; kra P-61: natural k order")
s = sub(s, "                const int dim = 16 * (U >> 8) + (U & 15), tok = 16 * ((U >> 6) & 3) + 4 * ((U >> 4) & 3);",
        "                const int dim = 16 * (U >> 8) + pi16(U & 15), tok = 16 * ((U >> 6) & 3) + 4 * ((U >> 4) & 3);"
        "   // kra P-61")
p.write_text(s)

f = root / "g2_fwd_v8.cuh"
s = f.read_text()
sig = "template <bool VL, int MODE = 0, bool GE = true, bool SG = false, bool PF = false>"
s = sub(s, sig, sig[:-1] + ", bool KP = false>")
head, rest = s.split("g2_fwd_h_v8o4_kernel(FwdOParams p) {", 1)
body, tail = rest.split("// kra P-49: g2_fwd_h_v8o4_kernel with NVH", 1)
body = sub(body, """    auto write_h = [&](int kb, f4* ht) {
        #pragma unroll
        for (int vt = 0; vt < 2; ++vt)
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                hx[vt][e] = bf_rne(ht[vt][e]);
                reinterpret_cast<short*>(hB)[(16 * vt + m) * kPadK + 16 * kb + c_col(l, e)] = hx[vt][e];
            }
    };""", """    auto write_h = [&](int kb, f4* ht) {
        #pragma unroll
        for (int vt = 0; vt < 2; ++vt) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                hx[vt][e] = bf_rne(ht[vt][e]);
                if constexpr (!KP)
                    reinterpret_cast<short*>(hB)[(16 * vt + m) * kPadK + 16 * kb + c_col(l, e)] = hx[vt][e];
            }
            if constexpr (KP)   // kra P-61: lane holds k = 4 q4 + e (consecutive)
                *reinterpret_cast<v4bh*>(hB + (16 * vt + m) * kPadK + 16 * kb + 4 * q4) = hx[vt];
        }
    };""")
s = head + "g2_fwd_h_v8o4_kernel(FwdOParams p) {" + body + "// kra P-49: g2_fwd_h_v8o4_kernel with NVH" + tail
for o, n in [
    ("case 1: if (sg) launch4(g2_fwd_h_v8o4_kernel<VL, 1, true, true, true>);",
     "case 1: if (sg) launch4(g2_fwd_h_v8o4_kernel<VL, 1, true, true, true, true>);"),
    ("                    else    launch4(g2_fwd_h_v8o4_kernel<VL, 1>);",
     "                    else    launch4(g2_fwd_h_v8o4_kernel<VL, 1, true, false, false, true>);"),
    ("case 2: launch4(g2_fwd_h_v8o4_kernel<VL, 2>); break;",
     "case 2: launch4(g2_fwd_h_v8o4_kernel<VL, 2, true, false, false, true>); break;"),
    ("default: launch4(g2_fwd_h_v8o4_kernel<VL, 0>); break;",
     "default: launch4(g2_fwd_h_v8o4_kernel<VL, 0, true, false, false, true>); break;"),
    ("    if (r1a && H >= g2_env_long(\"KDA_G2_V8O4W_MIN_H\", 40)",
     "    if (r1a && mode >= 3 && H >= g2_env_long(\"KDA_G2_V8O4W_MIN_H\", 40)"),   # kra P-61 probe scope
]:
    s = sub(s, o, n)
f.write_text(s)
print("P-61 APPLIED")
