"""kra P-61c (bitwise): write_h with ds_write_b64 in the chain-bound (SG / TP8) recurrence instances, kernel-local.

P-61/P-61b got the lane-permuted Kn and natural-k Gn from the producers; on bw7 that cost prep_a 3-7% (+44 us at
16K/12) plus two re-index copies, eating the recurrence gain.  Here nothing outside g2_fwd_h_v8o4_kernel changes:
  * Kn: lane (q, m) loads the 16 B pair of source lane (q, pi16(m)) -- same 1 KB group, so the load stays coalesced;
    GEMM2's C fragment then holds the consecutive k = 4 q + e and hB takes one 8-byte write per (v-tile, k-block);
  * Gn (pi16-permuted producer layout: Gn[16a + j] = decay[16a + pi16(j)]): the natural-k decays of the lane are
    Gn[16a + q + 4e], gathered as four dword loads (prefetched one step ahead like the f4 load they replace).
t-contraction order and every value unchanged -> bit-identical.  Used by the SG instances (modes 1 / 4).
    python3 p61c_kn_load_perm.py <tree>"""
import sys
from pathlib import Path

f = Path(sys.argv[1]) / "csrc" / "g2" / "g2_fwd_v8.cuh"
s = f.read_text()


def sub(s, old, new, cnt=1):
    assert s.count(old) == cnt, (old[:80], s.count(old))
    return s.replace(old, new)


sig = "template <bool VL, int MODE = 0, bool GE = true, bool SG = false, bool PF = false>"
s = sub(s, sig, sig[:-1] + ", bool KP = false>")
head, rest = s.split("g2_fwd_h_v8o4_kernel(FwdOParams p) {", 1)
body, tail = rest.split("// kra P-49: g2_fwd_h_v8o4_kernel with NVH", 1)
body = sub(body, """    auto loadG = [&](int i, int kb, f4& gl) {
        gl = *reinterpret_cast<const f4*>(p.Gn + ((chunk + i) * 8 + kb) * 16 + q4 * 4);
    };""", """    auto loadG = [&](int i, int kb, f4& gl) {
        if constexpr (KP) {   // kra P-61c: natural-k decays of this lane (k = 16 kb + 4 q4 + e)
            const float* g = p.Gn + ((chunk + i) * 8 + kb) * 16 + q4;
            #pragma unroll
            for (int e = 0; e < 4; ++e) gl[e] = g[4 * e];
        } else {
            gl = *reinterpret_cast<const f4*>(p.Gn + ((chunk + i) * 8 + kb) * 16 + q4 * 4);
        }
    };""")
body = sub(body, """            ld16x2(p.Kn + (((chunk + i) * 8 + kb) * 4) * 256 + j * 512 + l * 8, kf[2 * j], kf[2 * j + 1]);""",
           """            ld16x2(p.Kn + (((chunk + i) * 8 + kb) * 4) * 256 + j * 512
                   + (KP ? ((l & 48) | pi16(l & 15)) : l) * 8, kf[2 * j], kf[2 * j + 1]);   // kra P-61c""")
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
            if constexpr (KP)   // kra P-61c: lane holds k = 4 q4 + e (consecutive)
                *reinterpret_cast<v4bh*>(hB + (16 * vt + m) * kPadK + 16 * kb + 4 * q4) = hx[vt];
        }
    };""")
s = head + "g2_fwd_h_v8o4_kernel(FwdOParams p) {" + body + "// kra P-49: g2_fwd_h_v8o4_kernel with NVH" + tail
s = sub(s, "case 1: if (sg) launch4(g2_fwd_h_v8o4_kernel<VL, 1, true, true, true>);   // kra P-60",
        "case 1: if (sg) launch4(g2_fwd_h_v8o4_kernel<VL, 1, true, true, true, true>);   // kra P-60, P-61c")
s = sub(s, "                    else if (sg) launch4(g2_fwd_h_v8o4_kernel<VL, 4, true, true>);",
        "                    else if (sg) launch4(g2_fwd_h_v8o4_kernel<VL, 4, true, true, false, true>);   // kra P-61c")
f.write_text(s)
print("P-61c APPLIED")
