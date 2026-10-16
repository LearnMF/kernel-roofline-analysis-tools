"""kra P-77 (bitwise): fold the backward l2n_apply into its consumers (byte-ledger candidate 1).

results/kda-bytes-ledger-20261016: l2n_apply reads q, k (102 MB) and writes qn, kn (101 MB) at 16K/12, each read by
wu, wy_s4_split (s4b) and bwd_tail; 156 us (624 us at 16K/48).  With KDA_G2_L2FOLD (default on) the backward passes the
RAW q, k plus the forward-saved q_rstd / k_rstd and every consumer forms qn = bf_rne(bf2f(q) * rstd[tok*H + h]) at
its own load -- l2n_apply's exact expression (the multiply is not contractible: no add follows).  Same bytes for
the consumers (q/k are qn/kn's size), l2n_apply's launch and traffic disappear.  The fold is taken only on the route
whose every qn/kn consumer understands it: HIP tail (nofla or t5), S4 split (S4=1, P4=3), l2norm in kernel.
    python3 p77_l2fold.py <tree>"""
import sys
from pathlib import Path

R = Path(sys.argv[1])
G = R / "csrc" / "g2"


def sub(path, old, new, cnt=1):
    s = path.read_text()
    assert s.count(old) == cnt, (path.name, old[:90], s.count(old))
    path.write_text(s.replace(old, new))


FOLD = """__device__ __forceinline__ v4bh g2_l2fold(v4bh x, float r) {   // kra P-77: l2n_apply's expression
    #pragma unroll
    for (int e = 0; e < 4; ++e) x[e] = bf_rne(bf2f(x[e]) * r);
    return x;
}
"""
# ---- common helper
sub(G / "g2_common.h", "__device__ __forceinline__ size_t g2_rm_tok(long tok, int hh, int H, int D) {",
    FOLD + "__device__ __forceinline__ size_t g2_rm_tok(long tok, int hh, int H, int D) {")

# ---- wu: three q/k load sites
wu = G / "g2_wu.cuh"
sub(wu, """    int kp = 0;    // kra P-61e: Kn lane-permuted (H <= KPERM_MAX_H)
};""", """    int kp = 0;    // kra P-61e: Kn lane-permuted (H <= KPERM_MAX_H)
    const float* q_rstd = nullptr;   // kra P-77: non-null -> q / k are RAW, qn/kn formed at load
    const float* k_rstd = nullptr;
};""")
sub(wu, """g2_glb(p_.ctok), g2_glb(p_.clen), p_.stop, p_.kp};""",
    """g2_glb(p_.ctok), g2_glb(p_.clen), p_.stop, p_.kp, g2_glb(p_.q_rstd), g2_glb(p_.k_rstd)};   // kra P-77: + rstd""")
sub(wu, """            const v4bh kv = g2_row_ok<M>(j, cl) ? *reinterpret_cast<const v4bh*>(p.k + so)
                                                 : v4bh{0, 0, 0, 0};""",
    """            const v4bh kv = g2_row_ok<M>(j, cl)
                ? (p.k_rstd ? g2_l2fold(*reinterpret_cast<const v4bh*>(p.k + so), p.k_rstd[so >> 7])   // kra P-77
                            : *reinterpret_cast<const v4bh*>(p.k + so))
                : v4bh{0, 0, 0, 0};""")
sub(wu, """                const v4bh qv = g2_row_ok<M>(j, cl) ? *reinterpret_cast<const v4bh*>(p.q + so)
                                                     : v4bh{0, 0, 0, 0};""",
    """                const v4bh qv = g2_row_ok<M>(j, cl)
                    ? (p.q_rstd ? g2_l2fold(*reinterpret_cast<const v4bh*>(p.q + so), p.q_rstd[so >> 7])   // kra P-77
                                : *reinterpret_cast<const v4bh*>(p.q + so))
                    : v4bh{0, 0, 0, 0};""")
sub(wu, """                    const v4bh qv = g2_row_ok<M>(j, cl)
                        ? *reinterpret_cast<const v4bh*>(p.q + sbase + (size_t)j * rstride + vq)
                        : v4bh{0, 0, 0, 0};""",
    """                    const size_t soq = sbase + (size_t)j * rstride + vq;
                    const v4bh qv = g2_row_ok<M>(j, cl)
                        ? (p.q_rstd ? g2_l2fold(*reinterpret_cast<const v4bh*>(p.q + soq), p.q_rstd[soq >> 7])   // kra P-77
                                    : *reinterpret_cast<const v4bh*>(p.q + soq))
                        : v4bh{0, 0, 0, 0};""")
sub(wu, """                     int B, int T, int H, const int* ctok, const int* clen, hipStream_t s) {
    // T = chunk-padded rows (NT = T/64); with ctok/clen the row inputs are packed.""",
    """                     int B, int T, int H, const int* ctok, const int* clen,
                     const void* q_rstd, const void* k_rstd, hipStream_t s) {   // kra P-77: rstd
    // T = chunk-padded rows (NT = T/64); with ctok/clen the row inputs are packed.""")
sub(wu, """    p.kp = H <= g2::g2_env_long("KDA_G2_KPERM_MAX_H", 20);   // kra P-61e
""", """    p.kp = H <= g2::g2_env_long("KDA_G2_KPERM_MAX_H", 20);   // kra P-61e
    p.q_rstd = (const float*)q_rstd; p.k_rstd = (const float*)k_rstd;   // kra P-77
""")

# ---- s4b: kk / qq per kt
wy = G / "g2_wy.cuh"
sub(wy, """    const bf16* do_row;                 // kra P-20: row-major do, read when do_ (aN_do) is null
    const bf16* vnc;                    // kra P-22: C-form v_new (std order), read when v_new is null
};""", """    const bf16* do_row;                 // kra P-20: row-major do, read when do_ (aN_do) is null
    const bf16* vnc;                    // kra P-22: C-form v_new (std order), read when v_new is null
    const float* q_rstd = nullptr;      // kra P-77: non-null -> q_row / k are RAW, qn/kn formed at load
    const float* k_rstd = nullptr;
};""")
head, rest = wy.read_text().split("__device__ __forceinline__ void s4b_tail(", 1)
body, tail = rest.split("p3_wy_s4b_kernel(WyS4BParams p) {", 1)
o1 = """    const bf16* dwTg = p.dwT + g2_nat_blk(i, hh, H, 0, true, kD * kBT);

    f4 dak[4];"""
assert body.count(o1) == 1
body = body.replace(o1, """    const bf16* dwTg = p.dwT + g2_nat_blk(i, hh, H, 0, true, kD * kBT);
    // kra P-77: this lane's row rstd (row-major [T,H,128] element offset >> 7 = tok*H + h)
    const float rq_ = (p.q_rstd && rok) ? p.q_rstd[(cb >> 7) + (size_t)row * H] : 0.f;
    const float rk_ = (p.k_rstd && rok) ? p.k_rstd[(cb >> 7) + (size_t)row * H] : 0.f;

    f4 dak[4];""")
o2 = """        const v4bh kk = rok ? ld8(k_ + ro + kc) : v4bh{0, 0, 0, 0};
        const v4bh qq = rok ? ld8(q_ + ro + kc) : v4bh{0, 0, 0, 0};"""
assert body.count(o2) == 1
body = body.replace(o2, """        const v4bh kk = rok ? (p.k_rstd ? g2_l2fold(ld8(k_ + ro + kc), rk_) : ld8(k_ + ro + kc)) : v4bh{0, 0, 0, 0};   // kra P-77
        const v4bh qq = rok ? (p.q_rstd ? g2_l2fold(ld8(q_ + ro + kc), rq_) : ld8(q_ + ro + kc)) : v4bh{0, 0, 0, 0};""")
wy.write_text(head + "__device__ __forceinline__ void s4b_tail(" + body + "p3_wy_s4b_kernel(WyS4BParams p) {" + tail)
sub(wy, """                                const int* clen, const void* do_row, const void* dv2c,
                                const void* vnc, hipStream_t s) {
    // T = chunk-padded rows (NT = T/64); with ctok/clen the row tensors are packed.""",
    """                                const int* clen, const void* do_row, const void* dv2c,
                                const void* vnc, const void* q_rstd, const void* k_rstd,
                                hipStream_t s) {   // kra P-77: rstd
    // T = chunk-padded rows (NT = T/64); with ctok/clen the row tensors are packed.""")
sub(wy, """                       ctok, clen, (const bf16*)do_row, (const bf16*)vnc};
        if (ctok) hipLaunchKernelGGL(p3_wy_s4b_kernel<true>, dim3(NT, H), dim3(256), 0, s, pb);""",
    """                       ctok, clen, (const bf16*)do_row, (const bf16*)vnc};
        pb.q_rstd = (const float*)q_rstd; pb.k_rstd = (const float*)k_rstd;   // kra P-77
        if (ctok) hipLaunchKernelGGL(p3_wy_s4b_kernel<true>, dim3(NT, H), dim3(256), 0, s, pb);""")
sub(wy, """                            nullptr, nullptr, nullptr, s);
}""", """                            nullptr, nullptr, nullptr, nullptr, nullptr, s);
}""")

# ---- bwd_tail: y = qn / kn formed from the raw q / k with the row's rstd (already in r4)
tl = G / "g2_bwd_tail.cuh"
sub(tl, """    int gate_none; int beta_f32;
};""", """    int gate_none; int beta_f32;
    int raw_y = 0;                           // kra P-77: qn/kn slots carry RAW q/k
};""")
sub(tl, """                            yv[w2][u] = *reinterpret_cast<const short2*>(y + so + u * (p.H * kK) + c2);
                            r4[w2][u] = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(rsl[w2]), rb + u));""",
    """                            yv[w2][u] = *reinterpret_cast<const short2*>(y + so + u * (p.H * kK) + c2);
                            r4[w2][u] = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(rsl[w2]), rb + u));
                            if (p.raw_y) {   // kra P-77: l2n_apply's expression with the same rstd
                                yv[w2][u].x = bf_rne(bf2f(yv[w2][u].x) * r4[w2][u]);
                                yv[w2][u].y = bf_rne(bf2f(yv[w2][u].y) * r4[w2][u]);
                            }""")
sub(tl, """                           int gate_none, int beta_f32, hipStream_t s) {
    // Dense: T rows, NT = T/64.  Varlen phase 2: T packed rows, NTv chunks (ctok/clen).""",
    """                           int gate_none, int beta_f32, int raw_y, hipStream_t s) {   // kra P-77: raw_y
    // Dense: T rows, NT = T/64.  Varlen phase 2: T packed rows, NTv chunks (ctok/clen).""")
sub(tl, """                     use_l2, use_beta, H, NT, ctok, clen, gate_none, beta_f32};
""", """                     use_l2, use_beta, H, NT, ctok, clen, gate_none, beta_f32};
    p.raw_y = raw_y && use_l2;   // kra P-77
""")

# ---- bindings
bd = G / "g2_bindings.cpp"
sub(bd, """                     void* kgB, void* vB,
                     int B, int T, int H, const int* ctok, const int* clen, hipStream_t s);""",
    """                     void* kgB, void* vB,
                     int B, int T, int H, const int* ctok, const int* clen,
                     const void* q_rstd, const void* k_rstd, hipStream_t s);   // kra P-77""")
sub(bd, """                           int gate_none, int beta_f32, hipStream_t s);""",
    """                           int gate_none, int beta_f32, int raw_y, hipStream_t s);   // kra P-77""")
sub(bd, """                                const int* clen, const void* do_row, const void* dv2c,
                                const void* vnc, hipStream_t s);""",
    """                                const int* clen, const void* do_row, const void* dv2c,
                                const void* vnc, const void* q_rstd, const void* k_rstd,
                                hipStream_t s);   // kra P-77""")
# wu wrapper
sub(bd, """                              c10::optional<torch::Tensor> clen, bool emit_p1, bool emit_vb) {
    TORCH_CHECK(k.dim() == 4 && k.size(0) == 1, "g2_wu: B=1 only (batch = varlen rows)");""",
    """                              c10::optional<torch::Tensor> clen, bool emit_p1, bool emit_vb,
                              c10::optional<torch::Tensor> q_rstd, c10::optional<torch::Tensor> k_rstd) {
    TORCH_CHECK(k.dim() == 4 && k.size(0) == 1, "g2_wu: B=1 only (batch = varlen rows)");
    const bool fold = q_rstd.has_value() && q_rstd->defined();   // kra P-77
    if (fold) {
        TORCH_CHECK(k_rstd.has_value() && k_rstd->defined(), "g2_wu: q_rstd without k_rstd");
        check_contig(*q_rstd, "wu q_rstd"); check_contig(*k_rstd, "wu k_rstd");
        check_f32(*q_rstd, "wu q_rstd"); check_f32(*k_rstd, "wu k_rstd");
    }""")
sub(bd, """                         1, (int)(chunks * 64), (int)heads, rm.ctok, rm.clen, cur_stream());
    TORCH_CHECK(rc == 0, "g2_wu failed: ", rc);""",
    """                         1, (int)(chunks * 64), (int)heads, rm.ctok, rm.clen,
                         fold ? q_rstd->data_ptr() : nullptr, fold ? k_rstd->data_ptr() : nullptr,
                         cur_stream());
    TORCH_CHECK(rc == 0, "g2_wu failed: ", rc);""")
sub(bd, """          py::arg("clen") = py::none(), py::arg("emit_p1") = true, py::arg("emit_vb") = true);""",
    """          py::arg("clen") = py::none(), py::arg("emit_p1") = true, py::arg("emit_vb") = true,
          py::arg("q_rstd") = py::none(), py::arg("k_rstd") = py::none());   // kra P-77""")
# bwd_tail wrapper
sub(bd, """                                    c10::optional<torch::Tensor> clen, bool gate_none) {
    // gate_none (F1, precomputed-g contract): dg_raw = fp32 reverse chunk-cumsum of dg,""",
    """                                    c10::optional<torch::Tensor> clen, bool gate_none, bool raw_y) {
    // gate_none (F1, precomputed-g contract): dg_raw = fp32 reverse chunk-cumsum of dg,""")
sub(bd, """                               gate_none ? 1 : 0, beta_f32 ? 1 : 0, cur_stream());
    TORCH_CHECK(rc == 0, "g2_bwd_tail failed: ", rc);""",
    """                               gate_none ? 1 : 0, beta_f32 ? 1 : 0, raw_y ? 1 : 0, cur_stream());
    TORCH_CHECK(rc == 0, "g2_bwd_tail failed: ", rc);""")
# wy_s4_split wrapper
sub(bd, """                                       c10::optional<torch::Tensor> vn_c,
                                       c10::optional<torch::Tensor> dv2_c) {""",
    """                                       c10::optional<torch::Tensor> vn_c,
                                       c10::optional<torch::Tensor> dv2_c,
                                       c10::optional<torch::Tensor> q_rstd,
                                       c10::optional<torch::Tensor> k_rstd) {   // kra P-77""")
sub(bd, """                                  has_vn ? nullptr : vn_c->data_ptr(), cur_stream());""",
    """                                  has_vn ? nullptr : vn_c->data_ptr(),
                                  (q_rstd.has_value() && q_rstd->defined()) ? q_rstd->data_ptr() : nullptr,
                                  (k_rstd.has_value() && k_rstd->defined()) ? k_rstd->data_ptr() : nullptr,
                                  cur_stream());""")

# ---- Python wrappers
po = R / "hip_kda" / "ops" / "g2_ops.py"
sub(po, """def wu(k, v, q, beta, A, g, heads, chunks, rowmap=None, emit_p1=True, emit_vb=True):""",
    """def wu(k, v, q, beta, A, g, heads, chunks, rowmap=None, emit_p1=True, emit_vb=True,
       q_rstd=None, k_rstd=None):""")
sub(po, """                     *_rowmap(rowmap)[:2], bool(emit_p1), bool(emit_vb))""",
    """                     *_rowmap(rowmap)[:2], bool(emit_p1), bool(emit_vb),
                     None if q_rstd is None else q_rstd.contiguous(),
                     None if k_rstd is None else k_rstd.contiguous())   # kra P-77: raw q/k + rstd""")
sub(po, """             beta_raw, lower_bound, beta_scale, use_l2, use_beta, rowmap=None, gate_none=False):""",
    """             beta_raw, lower_bound, beta_scale, use_l2, use_beta, rowmap=None, gate_none=False,
             raw_y=False):""")
sub(po, """                           bool(use_l2), bool(use_beta), *_rowmap(rowmap)[:2], bool(gate_none))""",
    """                           bool(use_l2), bool(use_beta), *_rowmap(rowmap)[:2], bool(gate_none),
                           bool(raw_y))   # kra P-77: qn/kn slots carry raw q/k""")
sub(po, """                q_row, v_row, dAqk, heads, tokens, scale, emit_dA=False, rowmap=None, do_row=None,
                vn_c=None, dv2_c=None):""",
    """                q_row, v_row, dAqk, heads, tokens, scale, emit_dA=False, rowmap=None, do_row=None,
                vn_c=None, dv2_c=None, q_rstd=None, k_rstd=None):""")
sub(po, """                              None if dv2_c is None else dv2_c.reshape(-1).contiguous())""",
    """                              None if dv2_c is None else dv2_c.reshape(-1).contiguous(),
                              None if q_rstd is None else q_rstd.reshape(-1).contiguous(),
                              None if k_rstd is None else k_rstd.reshape(-1).contiguous())   # kra P-77""")

# ---- k3_g2 backward
k3 = R / "hip_kda" / "ops" / "k3_g2.py"
sub(k3, """        qn, kn = q, k
        if ctx.use_qk_l2norm_in_kernel:
            if os.environ.get("KDA_G2_BL2", "1").strip() != "0":""",
    """        qn, kn = q, k
        # kra P-77: l2n folded into wu / s4b / bwd_tail (raw q/k + rstd) on the route whose every
        # qn/kn consumer understands it; KDA_G2_L2FOLD=0 restores the l2n_apply launch.
        _env = lambda n, d: os.environ.get(n, d).strip()
        t5_early = (_env("KDA_G2_T5", "1") != "0" and not ctx.precomputed_g
                    and g_raw.dtype == torch.bfloat16 and a_log.dtype == torch.float32
                    and dt_bias.dtype == torch.float32
                    and (not ctx.use_beta_sigmoid_in_kernel or beta_raw.dtype == torch.bfloat16))
        l2fold = (ctx.use_qk_l2norm_in_kernel and _env("KDA_G2_L2FOLD", "1") != "0"
                  and _env("KDA_G2_BL2", "1") != "0" and _env("KDA_G2_S4", "1") != "0"
                  and _env("KDA_G2_P4", "3") == "3" and (hipi or t5_early))
        fq_r, fk_r = (q_rstd, k_rstd) if l2fold else (None, None)
        if ctx.use_qk_l2norm_in_kernel and not l2fold:
            if os.environ.get("KDA_G2_BL2", "1").strip() != "0":""")
sub(k3, """        Wn, Wb, Un, KgA, Kn, Qn, kgB, vB = g2_ops.wu(kn, v, qn, beta, Akk, g, H, NT, rowmap=lay,
                                                     emit_p1=not kept, emit_vb=not no_vb)""",
    """        Wn, Wb, Un, KgA, Kn, Qn, kgB, vB = g2_ops.wu(kn, v, qn, beta, Akk, g, H, NT, rowmap=lay,
                                                     emit_p1=not kept, emit_vb=not no_vb,
                                                     q_rstd=fq_r, k_rstd=fk_r)   # kra P-77""")
sub(k3, """                                                             do_row=do if no_ando else None,
                                                             vn_c=vn_c, dv2_c=dv2_c)""",
    """                                                             do_row=do if no_ando else None,
                                                             vn_c=vn_c, dv2_c=dv2_c,
                                                             q_rstd=fq_r, k_rstd=fk_r)   # kra P-77""")
sub(k3, """                ctx.use_qk_l2norm_in_kernel, ctx.use_beta_sigmoid_in_kernel, rowmap=lay,
                gate_none=pg)""", """                ctx.use_qk_l2norm_in_kernel, ctx.use_beta_sigmoid_in_kernel, rowmap=lay,
                gate_none=pg, raw_y=l2fold)   # kra P-77""")
sub(k3, """                beta_raw, lower_bound, 2.0 if ctx.allow_neg_eigval else 1.0,
                ctx.use_qk_l2norm_in_kernel, ctx.use_beta_sigmoid_in_kernel, rowmap=lay)""",
    """                beta_raw, lower_bound, 2.0 if ctx.allow_neg_eigval else 1.0,
                ctx.use_qk_l2norm_in_kernel, ctx.use_beta_sigmoid_in_kernel, rowmap=lay,
                raw_y=l2fold)   # kra P-77""")
print("P-77 APPLIED")
