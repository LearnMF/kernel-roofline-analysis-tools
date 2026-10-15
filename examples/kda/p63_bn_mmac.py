"""kra P-63 probe (bitwise, on top of P-62): the recompute / keep-state recurrence (kp, modes 2/3/4) emits bN(h)
from registers through one selector MMAC per (v-tile, k-block) -- no 16 ds_read_u16 of hB after barrier A.

bN slot (kt, vt, lane (q, m), e) = h[16 kt + pi16(m)][16 vt + 4 q + e].  With KP the bf16 state hx[vt] of a lane is
B-fragment shaped (row v, k = 4q + e).  If GEMM2 runs with its A rows (vB) pi16-relabelled, row v of lane n is pi16(n)
(write_h stores it at hB row pi16(n), so hB stays natural); then
    C = MMAC(A = S, B = hx),  S[m'][k] = [k == pi16(m')]
gives lane (q, m') element e = h[k = pi16(m')][v = pi16(q + 4e) = 4q + e] -- the bN slot.  C starts at -0.0 so every
sum is x + (+-0) terms = x exactly (also for x = -0), and the bf16 round trip of an exact bf16 value is the identity.
The emission moves from after barrier A (reading hB) to right after write_h (registers).
    python3 p63_bn_mmac.py <tree>"""
import sys
from pathlib import Path

f = Path(sys.argv[1]) / "csrc" / "g2" / "g2_fwd_v8.cuh"
s = f.read_text()


def sub(s, old, new, cnt=1):
    assert s.count(old) == cnt, (old[:80], s.count(old))
    return s.replace(old, new)


sig = "template <bool VL, int MODE = 0, bool GE = true, bool SG = false, bool PF = false, bool KP = false, bool VP = false>"
s = sub(s, sig, sig[:-1] + ", bool BP = false>")
head, rest = s.split("g2_fwd_h_v8o4_kernel(FwdOParams p) {", 1)
body, tail = rest.split("// kra P-49: g2_fwd_h_v8o4_kernel with NVH", 1)
assert "KP" in body
# write_h: hB row pi16(m) when BP; emit bN from the registers right after (chunk ib, if ib >= 0)
body = sub(body, """    auto write_h = [&](int kb, f4* ht) {
        #pragma unroll
        for (int vt = 0; vt < 2; ++vt) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                hx[vt][e] = bf_rne(ht[vt][e]);
                if constexpr (!KP)
                    reinterpret_cast<short*>(hB)[(16 * vt + m) * kPadK + 16 * kb + c_col(l, e)] = hx[vt][e];
            }
            if constexpr (KP)   // kra P-61e: Kn lane-permuted -> lane holds k = 4 q4 + e (consecutive)
                *reinterpret_cast<v4bh*>(hB + (16 * vt + m) * kPadK + 16 * kb + 4 * q4) = hx[vt];
        }
    };""", """    // kra P-63: selector A fragment S[m][k] = [k == pi16(m)] (bf16 1.0 / 0), lane (q4, m) holds k = 4 q4 + j
    v4bh bsel;
    #pragma unroll
    for (int j = 0; j < 4; ++j) bsel[j] = (4 * q4 + j == pi16(m)) ? (short)0x3f80 : (short)0;
    const int hrow = BP ? pi16(m) : m;   // kra P-63: GEMM2 / hB row of this lane
    auto write_h = [&](int kb, f4* ht, int ib = -1) {
        #pragma unroll
        for (int vt = 0; vt < 2; ++vt) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                hx[vt][e] = bf_rne(ht[vt][e]);
                if constexpr (!KP)
                    reinterpret_cast<short*>(hB)[(16 * vt + m) * kPadK + 16 * kb + c_col(l, e)] = hx[vt][e];
            }
            if constexpr (KP)   // kra P-61e: Kn lane-permuted -> lane holds k = 4 q4 + e (consecutive)
                *reinterpret_cast<v4bh*>(hB + (16 * vt + hrow) * kPadK + 16 * kb + 4 * q4) = hx[vt];
            if constexpr (BP) {   // kra P-63: bN(h) slot straight from the registers
                if (hasBN && ib >= 0) {
                    const f4 nz = f4{-0.f, -0.f, -0.f, -0.f};
                    const f4 c = mmac(bsel, hx[vt], nz);
                    v4bh val;
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) val[e] = bf_rne(c[e]);
                    const size_t off = ((((size_t)ib * p.H + bh) * 8 + kb) * 8 + (2 * vs + vt)) * 256 + l * 4;
                    *reinterpret_cast<v4bh*>(p.HbN + off) = val;
                }
            }
        }
    };""")
body = sub(body, """    write_h(kb0, ht0); write_h(kb1, ht1);
    if constexpr (PF)""", """    write_h(kb0, ht0, c0); write_h(kb1, ht1, c0);   // kra P-63: BP emits bN(h_c0) here
    if constexpr (PF)""")
body = sub(body, """        bN_emit(i, kb0);
        bN_emit(i, kb1);""", """        if constexpr (!BP) {   // kra P-63: BP emitted bN(h_i) at the end of the previous step
            bN_emit(i, kb0);
            bN_emit(i, kb1);
        }""")
body = sub(body, """                const v4bh va = ld8(vB + (16 * vt + m) * kPadT + 16 * rt + q4 * 4);
                upd = mmac(va, kf0[rt], upd);""", """                const v4bh va = ld8(vB + (16 * vt + hrow) * kPadT + 16 * rt + q4 * 4);   // kra P-63
                upd = mmac(va, kf0[rt], upd);""")
body = sub(body, """                const v4bh va = ld8(vB + (16 * vt + m) * kPadT + 16 * rt + q4 * 4);
                upd = mmac(va, kf1[rt], upd);""", """                const v4bh va = ld8(vB + (16 * vt + hrow) * kPadT + 16 * rt + q4 * 4);   // kra P-63
                upd = mmac(va, kf1[rt], upd);""")
body = sub(body, """        write_h(kb0, ht0);
        write_h(kb1, ht1);
""", """        write_h(kb0, ht0, i + 1 < c1 ? i + 1 : -1);   // kra P-63: h_{i+1} -> bN(h_{i+1})
        write_h(kb1, ht1, i + 1 < c1 ? i + 1 : -1);
""")
s = head + "g2_fwd_h_v8o4_kernel(FwdOParams p) {" + body + "// kra P-49: g2_fwd_h_v8o4_kernel with NVH" + tail
for o, n in [
    ("case 2: if (kp) launch4(g2_fwd_h_v8o4_kernel<VL, 2, true, false, false, true, true>);",
     "case 2: if (kp) launch4(g2_fwd_h_v8o4_kernel<VL, 2, true, false, false, true, true, true>);"),
    ("case 3: if (kp) launch4(g2_fwd_h_v8o4_kernel<VL, 3, true, false, false, true>);",
     "case 3: if (kp) launch4(g2_fwd_h_v8o4_kernel<VL, 3, true, false, false, true, false, true>);"),
    ("case 4: if (raw4) { if (kp) launch4raw(g2_fwd_h_v8o4_kernel<VL, 4, false, false, false, true>);",
     "case 4: if (raw4) { if (kp) launch4raw(g2_fwd_h_v8o4_kernel<VL, 4, false, false, false, true, false, true>);"),
    ("                    else if (sg) { if (kp) launch4(g2_fwd_h_v8o4_kernel<VL, 4, true, true, false, true>);",
     "                    else if (sg) { if (kp) launch4(g2_fwd_h_v8o4_kernel<VL, 4, true, true, false, true, false, true>);"),
    ("                    else if (kp) launch4(g2_fwd_h_v8o4_kernel<VL, 4, true, false, false, true>);",
     "                    else if (kp) launch4(g2_fwd_h_v8o4_kernel<VL, 4, true, false, false, true, false, true>);"),
]:
    s = sub(s, o, n)
f.write_text(s)
print("P-63 APPLIED")
