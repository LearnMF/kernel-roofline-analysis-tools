"""kra P-59 probe (bitwise): the forward recurrence computes v_new transposed so its LDS hand-off is one
ds_write_b64 per tile instead of four ds_write_b16.

GEMM1 runs operand-swapped (C' = h^T W^T, bit-identical to the transpose).  Lane (q4, m) of C' then holds
v = m and t = B-lane (q4 + 4e); prep stores Wn with its 16 lanes pi16-permuted (store address only), so B-lane
n carries W row pi16(n) and the lane's four elements are the consecutive t = 4 q4 + e -> one 8-byte write into
vB[v][t].  U must match C' elementwise: prep computes it with the swapped MMAC against pi16-shuffled Ai rows
(Un lane (q4, m) = U[4 q4 + e][m]).  Contraction order is untouched everywhere -> outputs bit-identical.
Forward modes (0/1/2) only; the backward recompute (modes 3/4) reads wu's Wn/Un and keeps the old path.
MODE 2 rebuilds the old-layout Vn/ANv fragments from vB after the vB barrier.  Probe scope: v8o4 only (the
wide v8o4w form is not selected for forward modes in this tree).
    python3 p59_vt.py <tree>"""
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "csrc" / "g2"


def sub(s, old, new, cnt=1):
    assert s.count(old) == cnt, (old[:80], s.count(old))
    return s.replace(old, new)


# ---------------------------------------------------------------- prep_b: Wn lane-permuted, Un transposed
p = root / "g2_fwd_prep.cuh"
s = p.read_text()
s = sub(s, """            const v4bh a = pack4(Ai[j]);
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) {
                const int n = pass == 0 ? pm : m;
                acc[vt] = mmac(a, *reinterpret_cast<const v4bh*>(T2 + (16 * vt + n) * kT2
                                                               + 16 * j + 4 * q4), acc[vt]);
            }""", """            const v4bh a = pack4(Ai[j]);
            const v4bh ap = pass == 1 ? g2_shfl4(a, 16 * q4 + pm) : a;   // kra P-59: lane m <- Ai row pi16(m)
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) {
                const int n = pass == 0 ? pm : m;
                const v4bh b = *reinterpret_cast<const v4bh*>(T2 + (16 * vt + n) * kT2 + 16 * j + 4 * q4);
                acc[vt] = pass == 0 ? mmac(a, b, acc[vt]) : mmac(b, ap, acc[vt]);   // kra P-59: U transposed
            }""")
s = sub(s, """            st16x2(out + nblk + (size_t)(2 * j + 8 * I) * 256 + 8 * l, pack4(acc[2 * j]), pack4(acc[2 * j + 1]));""",
        """            st16x2(out + nblk + (size_t)(2 * j + 8 * I) * 256 + 8 * (pass == 0 ? 16 * q4 + pm : l),   // kra P-59
                   pack4(acc[2 * j]), pack4(acc[2 * j + 1]));""")
p.write_text(s)

c = root / "g2_common.h"
s = c.read_text()
s = sub(s, "__device__ __forceinline__ int pi16(int n) { return 4 * (n & 3) + (n >> 2); }",
        """__device__ __forceinline__ int pi16(int n) { return 4 * (n & 3) + (n >> 2); }
// kra P-59: a lane's 4 bf16 fetched from lane src
__device__ __forceinline__ v4bh g2_shfl4(v4bh x, int src) {
    int w[2];
    __builtin_memcpy(w, &x, 8);
    w[0] = __shfl(w[0], src, 64);
    w[1] = __shfl(w[1], src, 64);
    v4bh y;
    __builtin_memcpy(&y, w, 8);
    return y;
}""")
c.write_text(s)

# ---------------------------------------------------------------- v8o4: swapped GEMM1, b64 vB writes
f = root / "g2_fwd_v8.cuh"
s = f.read_text()
sig = "template <bool VL, int MODE = 0, bool GE = true, bool SG = false>"
s = sub(s, sig, sig[:-1] + ", bool TV = false>")
head, rest = s.split("g2_fwd_h_v8o4_kernel(FwdOParams p) {", 1)
body, tail = rest.split("// kra P-49: g2_fwd_h_v8o4_kernel with NVH", 1)
body = sub(body, """            acc0 = mmac(wf[kt], hbf0[kt], acc0);
            acc1 = mmac(wf[kt], hbf1[kt], acc1);""",
           """            acc0 = TV ? mmac(hbf0[kt], wf[kt], acc0) : mmac(wf[kt], hbf0[kt], acc0);   // kra P-59
            acc1 = TV ? mmac(hbf1[kt], wf[kt], acc1) : mmac(wf[kt], hbf1[kt], acc1);""")
body = sub(body, """                reinterpret_cast<short*>(vB)[(16 * 0 + c_col(l, e)) * kPadT + 16 * r1 + m] = x0[e];
                reinterpret_cast<short*>(vB)[(16 * 1 + c_col(l, e)) * kPadT + 16 * r1 + m] = x1[e];
            }""", """                if constexpr (!TV) {
                    reinterpret_cast<short*>(vB)[(16 * 0 + c_col(l, e)) * kPadT + 16 * r1 + m] = x0[e];
                    reinterpret_cast<short*>(vB)[(16 * 1 + c_col(l, e)) * kPadT + 16 * r1 + m] = x1[e];
                }
            }
            if constexpr (TV) {   // kra P-59: lane holds v = m, t = 4 q4 + e (consecutive)
                *reinterpret_cast<v4bh*>(vB + (16 * 0 + m) * kPadT + 16 * r1 + 4 * q4) = x0;
                *reinterpret_cast<v4bh*>(vB + (16 * 1 + m) * kPadT + 16 * r1 + 4 * q4) = x1;
            }""")
vn_start = "            if (hasVn) {                       // forward: Vn unused (null)"
i0 = body.index(vn_start)
i1 = body.index("\n        }\n", body.index("if (hasAN)", i0)) + 1
vn_block = body[i0:i1]
body = body[:i0] + "            if constexpr (!TV) {\n" + vn_block + "            }\n" + body[i1:]
bar = "        lds_barrier();                                   // vB (v_new_i) ready\n"
body = sub(body, bar, bar + """        if constexpr (TV) if (hasVn || hasAN) {   // kra P-59: old-layout Vn/ANv fragments from vB
            v4bh x0, x1;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                x0[e] = reinterpret_cast<const short*>(vB)[(16 * 0 + c_col(l, e)) * kPadT + 16 * r1 + m];
                x1[e] = reinterpret_cast<const short*>(vB)[(16 * 1 + c_col(l, e)) * kPadT + 16 * r1 + m];
            }
""" + vn_block + "        }\n")
s = head + "g2_fwd_h_v8o4_kernel(FwdOParams p) {" + body + "// kra P-49: g2_fwd_h_v8o4_kernel with NVH" + tail
for o, n in [
    ("case 1: if (sg) launch4(g2_fwd_h_v8o4_kernel<VL, 1, true, true>);",
     "case 1: if (sg) launch4(g2_fwd_h_v8o4_kernel<VL, 1, true, true, true>);"),
    ("                    else    launch4(g2_fwd_h_v8o4_kernel<VL, 1>);",
     "                    else    launch4(g2_fwd_h_v8o4_kernel<VL, 1, true, false, true>);"),
    ("case 2: launch4(g2_fwd_h_v8o4_kernel<VL, 2>); break;",
     "case 2: launch4(g2_fwd_h_v8o4_kernel<VL, 2, true, false, true>); break;"),
    ("default: launch4(g2_fwd_h_v8o4_kernel<VL, 0>); break;",
     "default: launch4(g2_fwd_h_v8o4_kernel<VL, 0, true, false, true>); break;"),
    ("    if (r1a && H >= g2_env_long(\"KDA_G2_V8O4W_MIN_H\", 40)",
     "    if (r1a && mode >= 3 && H >= g2_env_long(\"KDA_G2_V8O4W_MIN_H\", 40)"),   # kra P-59 probe scope
]:
    s = sub(s, o, n)
f.write_text(s)
print("P-59 APPLIED")
