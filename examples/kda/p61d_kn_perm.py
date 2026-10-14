"""kra P-61d (bitwise, mergeable form of P-61b): when H <= KDA_G2_KPERM_MAX_H (default 20; the TP8 heads) the
recurrence stores h with one ds_write_b64 per (v-tile, k-block) instead of four ds_write_b16.

  * producers emit Kn with the 16 lanes of every sub-block pi16-permuted by permuting the STORE address only
    (prep_a reads its T2 tile exactly as before -- P-61b's permuted T2 reads cost prep_a 3-7%; wu likewise);
  * prep_a writes the forward Gn in natural k order;
  * the backward replaces exp2_(Gn_raw.clone()) (copy + exp2) by ONE kernel writing the permuted Gn (dhu) and
    the natural-k Gn (recompute); the deep-saturation raw instance gets a natural Gn_raw only when it runs;
  * the non-default forward paths (X1 / fwd_intra2 / FLA scaffold) re-index Kn / Gn in Python;
  * g2_fwd_h_v8o4_kernel gets KP (write_h b64) instances for every mode when kp; v8o4w / the 8-wave form are not
    taken when kp.
Every value and every contraction order unchanged -> bit-identical.
    python3 p61d_kn_perm.py <tree>"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
g2 = root / "csrc" / "g2"


def sub(s, old, new, cnt=1):
    assert s.count(old) == cnt, (old[:80], s.count(old))
    return s.replace(old, new)


ENV = 'g2_env_long("KDA_G2_KPERM_MAX_H", 20)'
ENVQ = 'g2::' + ENV                      # host functions outside namespace g2

# ---------------------------------------------------------------- prep_a
p = g2 / "g2_fwd_prep.cuh"
s = p.read_text()
s = sub(s, """    int stop;                           // phase timing (G2_PREP_STOP): 0 = full
};""", """    int stop;                           // phase timing (G2_PREP_STOP): 0 = full
    int kp = 0;                         // kra P-61d: Kn lane-permuted + Gn natural-k (H <= KPERM_MAX_H)
};""")
s = sub(s, "                    gnp[(col >> 4) * 16 + (col & 3) * 4 + ((col & 15) >> 2)] = exp2f(refv[e]);   // kra P-52",
        "                    gnp[p.kp ? col : (col >> 4) * 16 + (col & 3) * 4 + ((col & 15) >> 2)] = exp2f(refv[e]);"
        "   // kra P-52, P-61d")
s = sub(s, "            st16x2(p.Kn + nblk + 8 * (size_t)P, h[0], h[1]);",
        "            st16x2(p.Kn + nblk + 8 * (size_t)(p.kp ? ((P & ~15) | pi16(P & 15)) : P), h[0], h[1]);   // kra P-61d")
s = sub(s, """    if (const char* e = getenv("G2_PREP_STOP")) pa.stop = atoi(e);""",
        """    if (const char* e = getenv("G2_PREP_STOP")) pa.stop = atoi(e);
    pa.kp = H <= """ + ENVQ + """;   // kra P-61d""")
p.write_text(s)

# ---------------------------------------------------------------- wu
w = g2 / "g2_wu.cuh"
s = w.read_text()
s = sub(s, """    int stop;      // T2 staged timing (G2_WU_STOP): 0 = full kernel; 1..9 return
                   // right after the named stage (uniform branch, no data effect)
};""", """    int stop;      // T2 staged timing (G2_WU_STOP): 0 = full kernel; 1..9 return
                   // right after the named stage (uniform branch, no data effect)
    int kp = 0;    // kra P-61d: Kn lane-permuted (H <= KPERM_MAX_H)
};""")
s = sub(s, "template <bool VL>\n__global__ void __launch_bounds__(256) g2_wu_kernel(WuParams p_) {",
        """// kra P-61d: wu_emit16 with the STORE lane pi16-permuted inside every 16-lane group (tile reads unchanged)
__device__ __forceinline__ void wu_emit16s(short* dst, const short* tile, int tid) {
    for (int t = tid; t < 1024; t += 256) {
        const int o0 = 512 * (t >> 6) + 4 * (t & 63);
        st16x2(dst + 8 * (size_t)((t & ~15) | pi16(t & 15)), *reinterpret_cast<const v4bh*>(tile + wu_tile_idx(o0)),
               *reinterpret_cast<const v4bh*>(tile + wu_tile_idx(o0 + 256)));
    }
}

template <bool VL>
__global__ void __launch_bounds__(256) g2_wu_kernel(WuParams p_) {""")
s = sub(s, "p_.H, p_.NT, g2_glb(p_.ctok), g2_glb(p_.clen), p_.stop};",
        "p_.H, p_.NT, g2_glb(p_.ctok), g2_glb(p_.clen), p_.stop, p_.kp};")
s = sub(s, "                wu_emit16(p.Kn + nblk, tile, tid);",
        "                if (p.kp) wu_emit16s(p.Kn + nblk, tile, tid);   // kra P-61d\n"
        "                else      wu_emit16(p.Kn + nblk, tile, tid);")
s = sub(s, """                   H, T / 64, ctok, clen, stop};""", """                   H, T / 64, ctok, clen, stop};
    p.kp = H <= """ + ENVQ + """;   // kra P-61d""")
w.write_text(s)

# ---------------------------------------------------------------- recurrence
f = g2 / "g2_fwd_v8.cuh"
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
            if constexpr (KP)   // kra P-61d: Kn lane-permuted -> lane holds k = 4 q4 + e (consecutive)
                *reinterpret_cast<v4bh*>(hB + (16 * vt + m) * kPadK + 16 * kb + 4 * q4) = hx[vt];
        }
    };""")
s = head + "g2_fwd_h_v8o4_kernel(FwdOParams p) {" + body + "// kra P-49: g2_fwd_h_v8o4_kernel with NVH" + tail
s = sub(s, """extern "C" int g2_exp2_inplace(float* x, long n, hipStream_t s) {""",
        """// kra P-61d: Gn_raw (pi16-permuted k) -> exp2f into the permuted Gn AND the natural-k Gn in one pass
__global__ void __launch_bounds__(256) g2_exp2_dual_kernel(const float* __restrict__ x, float* __restrict__ yp,
                                                           float* __restrict__ yn, long n) {
    const long i = (long)blockIdx.x * 256 + threadIdx.x;
    if (i < n) {
        const float v = exp2f(x[i]);
        yp[i] = v;
        yn[(i & ~15L) | pi16((int)(i & 15))] = v;
    }
}
extern "C" int g2_exp2_dual(const float* x, float* yp, float* yn, long n, hipStream_t s) {
    if (n <= 0) return 0;
    hipLaunchKernelGGL(g2_exp2_dual_kernel, dim3((unsigned)((n + 255) / 256)), dim3(256), 0, s, x, yp, yn, n);
    return (int)hipGetLastError();
}
extern "C" int g2_exp2_inplace(float* x, long n, hipStream_t s) {""")
s = sub(s, """    if (const char* e = getenv("KDA_G2_R1A")) r1a = atoi(e);""",
        """    if (const char* e = getenv("KDA_G2_R1A")) r1a = atoi(e);
    const bool kp = H <= """ + ENV + """;   // kra P-61d: Kn lane-permuted / Gn natural-k producers
    if (kp) r1a = 1;""")
old_sw = """        switch (mode) {
            case 1: if (sg) launch4(g2_fwd_h_v8o4_kernel<VL, 1, true, true, true>);   // kra P-60
                    else    launch4(g2_fwd_h_v8o4_kernel<VL, 1>);
                    break;
            case 2: launch4(g2_fwd_h_v8o4_kernel<VL, 2>); break;
            case 3: launch4(g2_fwd_h_v8o4_kernel<VL, 3>); break;
            case 4: if (raw4) launch4raw(g2_fwd_h_v8o4_kernel<VL, 4, false>);
                    else if (sg) launch4(g2_fwd_h_v8o4_kernel<VL, 4, true, true>);
                    else      launch4(g2_fwd_h_v8o4_kernel<VL, 4>);
                    break;
            default: launch4(g2_fwd_h_v8o4_kernel<VL, 0>); break;
        }"""
new_sw = """        switch (mode) {   // kra P-61d: KP instances whenever the producers emit the kp layouts
            case 1: if (sg) { if (kp) launch4(g2_fwd_h_v8o4_kernel<VL, 1, true, true, true, true>);   // kra P-60
                              else    launch4(g2_fwd_h_v8o4_kernel<VL, 1, true, true, true>); }
                    else if (kp) launch4(g2_fwd_h_v8o4_kernel<VL, 1, true, false, false, true>);
                    else    launch4(g2_fwd_h_v8o4_kernel<VL, 1>);
                    break;
            case 2: if (kp) launch4(g2_fwd_h_v8o4_kernel<VL, 2, true, false, false, true>);
                    else    launch4(g2_fwd_h_v8o4_kernel<VL, 2>);
                    break;
            case 3: if (kp) launch4(g2_fwd_h_v8o4_kernel<VL, 3, true, false, false, true>);
                    else    launch4(g2_fwd_h_v8o4_kernel<VL, 3>);
                    break;
            case 4: if (raw4) { if (kp) launch4raw(g2_fwd_h_v8o4_kernel<VL, 4, false, false, false, true>);
                                else    launch4raw(g2_fwd_h_v8o4_kernel<VL, 4, false>); }
                    else if (sg) { if (kp) launch4(g2_fwd_h_v8o4_kernel<VL, 4, true, true, false, true>);
                                   else    launch4(g2_fwd_h_v8o4_kernel<VL, 4, true, true>); }
                    else if (kp) launch4(g2_fwd_h_v8o4_kernel<VL, 4, true, false, false, true>);
                    else      launch4(g2_fwd_h_v8o4_kernel<VL, 4>);
                    break;
            default: if (kp) launch4(g2_fwd_h_v8o4_kernel<VL, 0, true, false, false, true>);
                     else    launch4(g2_fwd_h_v8o4_kernel<VL, 0>);
                     break;
        }"""
s = sub(s, old_sw, new_sw)
s = sub(s, "    if (r1a && H >= g2_env_long(\"KDA_G2_V8O4W_MIN_H\", 40)",
        "    if (r1a && !kp && H >= g2_env_long(\"KDA_G2_V8O4W_MIN_H\", 40)")
f.write_text(s)

# ---------------------------------------------------------------- binding
b = g2 / "g2_bindings.cpp"
s = b.read_text()
s = sub(s, "extern \"C\" int g2_exp2_inplace(float* x, long n, hipStream_t s);   // kra P-52",
        "extern \"C\" int g2_exp2_inplace(float* x, long n, hipStream_t s);   // kra P-52\n"
        "extern \"C\" int g2_exp2_dual(const float* x, float* yp, float* yn, long n, hipStream_t s);   // kra P-61d")
s = sub(s, "// kra P-52: in-place device exp2f (Gn -> chunk decay factors)\ntorch::Tensor exp2_(torch::Tensor x) {",
        """// kra P-61d: Gn_raw -> (exp2f permuted Gn, exp2f natural-k Gn) in one pass
std::vector<torch::Tensor> exp2_dual(torch::Tensor x) {
    check_contig(x, "exp2_dual"); check_f32(x, "exp2_dual");
    TORCH_CHECK(x.numel() % 16 == 0, "exp2_dual: numel must be a multiple of 16");
    auto yp = torch::empty_like(x), yn = torch::empty_like(x);
    const int rc = g2_exp2_dual(x.data_ptr<float>(), yp.data_ptr<float>(), yn.data_ptr<float>(),
                                (long)x.numel(), cur_stream());
    TORCH_CHECK(rc == 0, "g2_exp2_dual failed: ", rc);
    return {yp, yn};
}

// kra P-52: in-place device exp2f (Gn -> chunk decay factors)
torch::Tensor exp2_(torch::Tensor x) {""")
s = sub(s, '    m.def("exp2_", &exp2_, "G2: in-place device exp2f (Gn -> decay factors) [kra P-52]");',
        '    m.def("exp2_", &exp2_, "G2: in-place device exp2f (Gn -> decay factors) [kra P-52]");\n'
        '    m.def("exp2_dual", &exp2_dual, "G2: exp2f into permuted + natural-k Gn [kra P-61d]");')
b.write_text(s)

# ---------------------------------------------------------------- python wiring
k = root / "hip_kda" / "ops" / "k3_g2.py"
s = k.read_text()
helpers = '''

# kra P-61d: H <= KDA_G2_KPERM_MAX_H -> producers emit Kn lane-permuted (lane n of every 16 carries pi16(n)) and the
# recurrence reads natural-k decays; C++ (prep / wu / fwd_h) applies the same rule from the same variable.
_PI16 = [4 * (n & 3) + (n >> 2) for n in range(16)]


def _kp(H):
    return H <= int(os.environ.get("KDA_G2_KPERM_MAX_H", "20"))


def _gn_nat(t):   # Gn [..][a][c][d] = k 16a+4d+c  ->  natural k order (a pure permutation)
    return t.view(-1, 8, 4, 4).transpose(-1, -2).contiguous().view(t.shape)


def _kn_perm(t):  # Kn: 16 B lane pairs, lane (q, m) <- lane (q, pi16(m)) inside every 512-element group
    idx = torch.tensor(_PI16, device=t.device)
    return t.view(-1, 4, 16, 8).index_select(2, idx).contiguous().view(t.shape)


def _raw4(H, seg):   # mirrors g2_fwd_h_v8o's deep-saturation raw instance choice
    nseg = 1 if seg is None else seg.numel() - 1
    return 4 * nseg * H >= int(os.environ.get("KDA_G2_BN_RAW_MIN_CTAS", "192"))
'''
anchor = "\nclass "
i = s.index(anchor)
s = s[:i] + helpers + s[i:]
s = sub(s, """        Gn = gn.to_gn(g, H) if lay is None else _to_gn_varlen(g, H, lay)
        # F2+F3: HIP (P1 v8).""", """        Gn = gn.to_gn(g, H) if lay is None else _to_gn_varlen(g, H, lay)
        if _kp(H):                                                # kra P-61d: kp layouts on this path too
            Kn, Gn = _kn_perm(Kn), _gn_nat(Gn)
        # F2+F3: HIP (P1 v8).""")
s = sub(s, """        Gn = g2_ops._ext().exp2_(Gn_raw.clone())
""", """        if _kp(H):   # kra P-61d: permuted Gn for dhu + natural-k Gn for the recompute, one pass
            Gn, Gn_rec = g2_ops._ext().exp2_dual(Gn_raw)
        else:
            Gn = g2_ops._ext().exp2_(Gn_raw.clone())
            Gn_rec = Gn
""")
s = sub(s, """            bN_h, Vn, aN_vn = g2_ops.fwd_h_bn(Wn, Kn, Un, Gn, seg=ctx.seg,
                                              emit_an=not no_an, gn_raw=Gn_raw)""",
        """            gn_raw_rec = _gn_nat(Gn_raw) if _kp(H) and _raw4(H, ctx.seg) else Gn_raw   # kra P-61d
            bN_h, Vn, aN_vn = g2_ops.fwd_h_bn(Wn, Kn, Un, Gn_rec, seg=ctx.seg,
                                              emit_an=not no_an, gn_raw=gn_raw_rec)""")
k.write_text(s)
print("P-61d APPLIED")
