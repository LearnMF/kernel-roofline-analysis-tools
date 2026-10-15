"""kra P-62 probe (bitwise, on top of P-61e): the forward recurrence (kp, modes 0/1/2) writes o straight from
registers -- no oStage LDS round trip.

Critical path of the production MODE 1 step after P-61e (16K/12): the bf16 o block goes through oStage (4
ds_write_b16 + wave lgkmcnt drain + ds_read_b64 per v1a) because the o C fragment gives lane (q4, m) the columns
v = q4 + 4e; ~300-450 of ~2700 cycles per step.  Here the v index of h / v_new / o is pi16-relabelled inside the
kernel, never in a contraction:
  * GEMM1 / oh read the B fragment (h) from hB row pi16(n) for lane n, the o pass reads vB row pi16(n) -> every C
    fragment holds the consecutive v = 4 q4 + e;
  * U must match GEMM1's C elementwise: prep_b emits Un with the pm trick Wn already uses (B row pi16(n)), when kp;
  * v_new goes to vB[v][t] at its true v row; GEMM2 (A = vB rows, natural) and hB are unchanged;
  * o: one 8-byte global store per v1a (bf16), or 4 consecutive fp32 (MODE 0 O);
  * MODE 2 rebuilds its old-layout Vn / ANv fragments from vB after the vB barrier;
  * the X1 / scaffold forward paths re-lay Un out in g2_kp_relayout (lane (q,m) element e <- lane (e,m) element q).
Contraction orders unchanged -> bit-identical.
    python3 p62_vperm.py <tree>"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
g2 = root / "csrc" / "g2"


def sub(s, old, new, cnt=1):
    assert s.count(old) == cnt, (old[:80], s.count(old))
    return s.replace(old, new)


# ---------------------------------------------------------------- prep_b: Un with the pm trick when kp
p = g2 / "g2_fwd_prep.cuh"
s = p.read_text()
s = sub(s, """    float scale;
    int H, NT;
};
""", """    float scale;
    int H, NT;
    int kp = 0;   // kra P-62: Un with the pm column order (as Wn) when H <= KPERM_MAX_H
};
""")
s = sub(s, "                const int n = pass == 0 ? pm : m;", "                const int n = (pass == 0 || p.kp) ? pm : m;   // kra P-62")
s = sub(s, """                           ctok, clen, (float)scale, H, NT};
""", """                           ctok, clen, (float)scale, H, NT};
    pb.kp = pa.kp;   // kra P-62
""")
p.write_text(s)

# ---------------------------------------------------------------- recurrence
f = g2 / "g2_fwd_v8.cuh"
s = f.read_text()
sig = "template <bool VL, int MODE = 0, bool GE = true, bool SG = false, bool PF = false, bool KP = false>"
s = sub(s, sig, sig[:-1] + ", bool VP = false>")
head, rest = s.split("g2_fwd_h_v8o4_kernel(FwdOParams p) {", 1)
body, tail = rest.split("// kra P-49: g2_fwd_h_v8o4_kernel with NVH", 1)
body = sub(body, """            hbf0[kt] = ld8(hB + (0 * 16 + m) * kPadK + 16 * kt + q4 * 4);
            hbf1[kt] = ld8(hB + (1 * 16 + m) * kPadK + 16 * kt + q4 * 4);""",
           """            hbf0[kt] = ld8(hB + (0 * 16 + (VP ? pi16(m) : m)) * kPadK + 16 * kt + q4 * 4);   // kra P-62
            hbf1[kt] = ld8(hB + (1 * 16 + (VP ? pi16(m) : m)) * kPadK + 16 * kt + q4 * 4);""")
body = sub(body, """                reinterpret_cast<short*>(vB)[(16 * 0 + c_col(l, e)) * kPadT + 16 * r1 + m] = x0[e];
                reinterpret_cast<short*>(vB)[(16 * 1 + c_col(l, e)) * kPadT + 16 * r1 + m] = x1[e];""",
           """                const int vc = VP ? 4 * q4 + e : c_col(l, e);   // kra P-62: lane's v columns
                reinterpret_cast<short*>(vB)[(16 * 0 + vc) * kPadT + 16 * r1 + m] = x0[e];
                reinterpret_cast<short*>(vB)[(16 * 1 + vc) * kPadT + 16 * r1 + m] = x1[e];""")
vn_start = "            if (hasVn) {                       // forward: Vn unused (null)"
i0 = body.index(vn_start)
i1 = body.index("\n        }\n", body.index("if (hasAN)", i0)) + 1
vn_block = body[i0:i1]
body = body[:i0] + "            if constexpr (!VP) {\n" + vn_block + "            }\n" + body[i1:]
bar = "        lds_barrier();                                   // vB (v_new_i) ready\n"
body = sub(body, bar, bar + """        if constexpr (VP) if (hasVn || hasAN) {   // kra P-62: old-layout Vn/ANv fragments from vB
            v4bh x0, x1;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                x0[e] = reinterpret_cast<const short*>(vB)[(16 * 0 + c_col(l, e)) * kPadT + 16 * r1 + m];
                x1[e] = reinterpret_cast<const short*>(vB)[(16 * 1 + c_col(l, e)) * kPadT + 16 * r1 + m];
            }
""" + vn_block + "        }\n")
body = sub(body, "                    oa = mmac(aqf[st], ld8(vB + (16 * v1a + m) * kPadT + 16 * st + q4 * 4), oa);",
           "                    oa = mmac(aqf[st], ld8(vB + (16 * v1a + (VP ? pi16(m) : m)) * kPadT + 16 * st + q4 * 4), oa);"
           "   // kra P-62")
body = sub(body, """                if (hasO) {
                    if (ook) {
                        #pragma unroll
                        for (int e = 0; e < 4; ++e)
                            p.O[obase + q4 + 4 * e] = o[e];
                    }
                } else {""", """                if constexpr (VP) {   // kra P-62: lane holds v = 4 q4 + e -> direct stores, no oStage
                    if (ook) {
                        if (hasO) {
                            *reinterpret_cast<f4*>(p.O + obase + 4 * q4) = o;
                        } else {
                            v4bh ov;
                            #pragma unroll
                            for (int e = 0; e < 4; ++e) ov[e] = bf_rne(o[e]);
                            *reinterpret_cast<v4bh*>(p.Ob + obase + 4 * q4) = ov;
                        }
                    }
                } else if (hasO) {
                    if (ook) {
                        #pragma unroll
                        for (int e = 0; e < 4; ++e)
                            p.O[obase + q4 + 4 * e] = o[e];
                    }
                } else {""")
s = head + "g2_fwd_h_v8o4_kernel(FwdOParams p) {" + body + "// kra P-49: g2_fwd_h_v8o4_kernel with NVH" + tail
for o, n in [
    ("if (kp) launch4(g2_fwd_h_v8o4_kernel<VL, 1, true, true, true, true>);",
     "if (kp) launch4(g2_fwd_h_v8o4_kernel<VL, 1, true, true, true, true, true>);"),
    ("else if (kp) launch4(g2_fwd_h_v8o4_kernel<VL, 1, true, false, false, true>);",
     "else if (kp) launch4(g2_fwd_h_v8o4_kernel<VL, 1, true, false, false, true, true>);"),
    ("case 2: if (kp) launch4(g2_fwd_h_v8o4_kernel<VL, 2, true, false, false, true>);",
     "case 2: if (kp) launch4(g2_fwd_h_v8o4_kernel<VL, 2, true, false, false, true, true>);"),
    ("default: if (kp) launch4(g2_fwd_h_v8o4_kernel<VL, 0, true, false, false, true>);",
     "default: if (kp) launch4(g2_fwd_h_v8o4_kernel<VL, 0, true, false, false, true, true>);"),
]:
    s = sub(s, o, n)
# kp_relayout: + Un (lane (q,m) element e <- lane (e,m) element q, per 256-element sub-block)
s = sub(s, """__global__ void __launch_bounds__(256) g2_kp_relayout_kernel(short* __restrict__ kn, long nkg,
                                                             float* __restrict__ gn, long ngg) {
    const long w = ((long)blockIdx.x * 256 + threadIdx.x) >> 6;
    const int l = threadIdx.x & 63;
    if (w < nkg) {""", """__global__ void __launch_bounds__(256) g2_kp_relayout_kernel(short* __restrict__ kn, long nkg,
                                                             float* __restrict__ gn, long ngg,
                                                             short* __restrict__ un) {
    const long w = ((long)blockIdx.x * 256 + threadIdx.x) >> 6;
    const int l = threadIdx.x & 63;
    if (un != nullptr && w < nkg) {   // kra P-62: Un -> pm column order (16 B pairs, nat16 inside 512)
        short v[8];
        #pragma unroll
        for (int sb = 0; sb < 2; ++sb)
            #pragma unroll
            for (int e = 0; e < 4; ++e)
                v[4 * sb + e] = un[512 * w + 8 * (16 * e + (l & 15)) + 4 * sb + (l >> 4)];
        v4bh a, b;
        #pragma unroll
        for (int e = 0; e < 4; ++e) { a[e] = v[e]; b[e] = v[4 + e]; }
        st16x2(un + 512 * w + 8 * l, a, b);
    }
    if (w < nkg) {""")
s = sub(s, """extern "C" int g2_kp_relayout(void* kn, long kn_elems, float* gn, long gn_elems, hipStream_t s) {""",
        """extern "C" int g2_kp_relayout(void* kn, long kn_elems, float* gn, long gn_elems, void* un, hipStream_t s) {""")
s = sub(s, """                       (short*)kn, nkg, gn, ngg);""", """                       (short*)kn, nkg, gn, ngg, (short*)un);""")
f.write_text(s)

b = g2 / "g2_bindings.cpp"
s = b.read_text()
s = sub(s, "extern \"C\" int g2_kp_relayout(void* kn, long kn_elems, float* gn, long gn_elems, hipStream_t s);",
        "extern \"C\" int g2_kp_relayout(void* kn, long kn_elems, float* gn, long gn_elems, void* un, hipStream_t s);")
s = sub(s, """void kp_relayout_(torch::Tensor kn, torch::Tensor gn) {
    check_contig(kn, "kp_relayout_"); check_contig(gn, "kp_relayout_"); check_f32(gn, "kp_relayout_");
    const int rc = g2_kp_relayout(kn.data_ptr(), (long)kn.numel(), gn.data_ptr<float>(), (long)gn.numel(),
                                  cur_stream());""", """void kp_relayout_(torch::Tensor kn, torch::Tensor gn, torch::Tensor un) {
    check_contig(kn, "kp_relayout_"); check_contig(gn, "kp_relayout_"); check_f32(gn, "kp_relayout_");
    check_contig(un, "kp_relayout_");
    TORCH_CHECK(un.numel() == kn.numel(), "kp_relayout_: Un / Kn sizes differ");
    const int rc = g2_kp_relayout(kn.data_ptr(), (long)kn.numel(), gn.data_ptr<float>(), (long)gn.numel(),
                                  un.data_ptr(), cur_stream());""")
b.write_text(s)

k = root / "hip_kda" / "ops" / "k3_g2.py"
s = k.read_text()
s = sub(s, """            Kn, Gn = Kn.contiguous(), Gn.contiguous()
            g2_ops._ext().kp_relayout_(Kn, Gn)""", """            Kn, Gn, Un = Kn.contiguous(), Gn.contiguous(), Un.contiguous()
            g2_ops._ext().kp_relayout_(Kn, Gn, Un)   # kra P-62: + Un""")
k.write_text(s)
print("P-62 APPLIED")
