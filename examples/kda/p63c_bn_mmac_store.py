"""kra P-63c probe (bitwise, on top of P-62 + P-63): the selector-MMAC bN values are computed at write_h (registers)
but STORED where bN_emit stored them (right after barrier A of the step that emits h_i).

P-63 put the HbN global stores right before barrier A; on gfx9 vmcnt counts stores in issue order, so the next K
prefetch wait also waited for those stores (R52; recompute 0.96x).  Here the stores keep their production position
in the issue stream; only the 16 ds_read_u16 + their LDS latency leave the step.
    python3 p63c_bn_mmac_store.py <tree>"""
import sys
from pathlib import Path

f = Path(sys.argv[1]) / "csrc" / "g2" / "g2_fwd_v8.cuh"
s = f.read_text()
head, rest = s.split("g2_fwd_h_v8o4_kernel(FwdOParams p) {", 1)
body, tail = rest.split("// kra P-49: g2_fwd_h_v8o4_kernel with NVH", 1)


def sub(b, old, new):
    assert b.count(old) == 1, (old[:80], b.count(old))
    return b.replace(old, new)


body = sub(body, """    auto write_h = [&](int kb, f4* ht, int ib = -1) {""", """    v4bh bnv[2][2];   // kra P-63c: bN(h) values [kb0/kb1][vt], stored after the next barrier A
    auto write_h = [&](int kb, f4* ht, int ib = -1) {""")
body = sub(body, """                if (hasBN && ib >= 0) {
                    const f4 nz = f4{-0.f, -0.f, -0.f, -0.f};
                    const f4 c = mmac(bsel, hx[vt], nz);
                    v4bh val;
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) val[e] = bf_rne(c[e]);
                    const size_t off = ((((size_t)ib * p.H + bh) * 8 + kb) * 8 + (2 * vs + vt)) * 256 + l * 4;
                    *reinterpret_cast<v4bh*>(p.HbN + off) = val;
                }""", """                if (hasBN && ib >= 0) {
                    const f4 nz = f4{-0.f, -0.f, -0.f, -0.f};
                    const f4 c = mmac(bsel, hx[vt], nz);
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) bnv[kb == kb0 ? 0 : 1][vt][e] = bf_rne(c[e]);
                }""")
body = sub(body, """        if constexpr (!BP) {   // kra P-63: BP emitted bN(h_i) at the end of the previous step
            bN_emit(i, kb0);
            bN_emit(i, kb1);
        }""", """        if constexpr (!BP) {   // kra P-63: BP emitted bN(h_i) at the end of the previous step
            bN_emit(i, kb0);
            bN_emit(i, kb1);
        } else if (hasBN) {    // kra P-63c: production store position, values from the registers
            #pragma unroll
            for (int kk = 0; kk < 2; ++kk)
                #pragma unroll
                for (int vt = 0; vt < 2; ++vt) {
                    const size_t off = ((((size_t)i * p.H + bh) * 8 + (kk ? kb1 : kb0)) * 8 + (2 * vs + vt)) * 256 + l * 4;
                    *reinterpret_cast<v4bh*>(p.HbN + off) = bnv[kk][vt];
                }
        }""")
f.write_text(head + "g2_fwd_h_v8o4_kernel(FwdOParams p) {" + body + "// kra P-49: g2_fwd_h_v8o4_kernel with NVH" + tail)
print("P-63c APPLIED")
