"""kra P-36 = fusion step F1 (A-class, must stay bit-identical): the backward no longer writes
qn/kn.  On the production S4 split route the three consumers (wu, s4b, bwd_tail) read the RAW q/k
plus the forward's saved q_rstd/k_rstd and apply bf16(x * rstd) on load -- the exact expression of
g2_l2n_apply_kernel -- so the l2n_apply launch and its 4 x T*H*128 bf16 of traffic disappear
(C1 group, 9.1% of launcher interface traffic; roadmap results/kda-gfx936-20261009/fusion).
ABI: new entry points g2_wu_l2n / g2_wy_s4_split_r2 (the old g2_wu / g2_wy_s4_split_r forward
to them with null rstd, so flash-train's runner keeps compiling); bwd_tail's int use_l2 = 2 means
"qn/kn hold raw q/k".  KDA_G2_F1=0 restores l2n_apply.
Usage: python3 p36_f1_l2n_on_load.py <tree root>"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
G2 = ROOT / "csrc" / "g2"


def sub(path, old, new, count=1):
    t = path.read_text()
    assert t.count(old) == count, (path.name, old[:90], t.count(old))
    path.write_text(t.replace(old, new))


def sub_in(path, fn_head, old, new):
    """replace exactly once inside the function that starts with fn_head"""
    t = path.read_text()
    i = t.index(fn_head)
    j = t.index("\n}\n", i)
    seg = t[i:j]
    assert seg.count(old) == 1, (path.name, fn_head, old[:80], seg.count(old))
    path.write_text(t[:i] + seg.replace(old, new) + t[j:])


# ------------------------------------------------------------------ wu
wu = G2 / "g2_wu.cuh"
sub(wu, """                   // right after the named stage (uniform branch, no data effect)
};""", """                   // right after the named stage (uniform branch, no data effect)
    const float* q_rstd; const float* k_rstd;   // kra F1: non-null -> q/k are RAW, l2norm on load
};""")
sub(wu, "g2_glb(p_.ctok), g2_glb(p_.clen), p_.stop};",
    "g2_glb(p_.ctok), g2_glb(p_.clen), p_.stop, g2_glb(p_.q_rstd), g2_glb(p_.k_rstd)};")
sub(wu, """            const v4bh kv = g2_row_ok<M>(j, cl) ? *reinterpret_cast<const v4bh*>(p.k + so)
                                                 : v4bh{0, 0, 0, 0};
            const f4 gf = ld16f(p.g + sbase + (size_t)g2_row_g<M>(j, cl) * rstride + vq);""",
    """            v4bh kv = g2_row_ok<M>(j, cl) ? *reinterpret_cast<const v4bh*>(p.k + so)
                                           : v4bh{0, 0, 0, 0};
            if (p.k_rstd != nullptr && g2_row_ok<M>(j, cl)) {   // kra F1: l2n_apply's value
                const float r_ = p.k_rstd[(size_t)(g2_row_tok<VL>(p.ctok, i) + j) * p.H + hh];
                #pragma unroll
                for (int e_ = 0; e_ < 4; ++e_) kv[e_] = bf_rne(bf2f(kv[e_]) * r_);
            }
            const f4 gf = ld16f(p.g + sbase + (size_t)g2_row_g<M>(j, cl) * rstride + vq);""")
sub(wu, """                const v4bh qv = g2_row_ok<M>(j, cl) ? *reinterpret_cast<const v4bh*>(p.q + so)
                                                     : v4bh{0, 0, 0, 0};""",
    """                v4bh qv = g2_row_ok<M>(j, cl) ? *reinterpret_cast<const v4bh*>(p.q + so)
                                               : v4bh{0, 0, 0, 0};
                if (p.q_rstd != nullptr && g2_row_ok<M>(j, cl)) {   // kra F1
                    const float r_ = p.q_rstd[(size_t)(g2_row_tok<VL>(p.ctok, i) + j) * p.H + hh];
                    #pragma unroll
                    for (int e_ = 0; e_ < 4; ++e_) qv[e_] = bf_rne(bf2f(qv[e_]) * r_);
                }""")
sub(wu, """                    const v4bh qv = g2_row_ok<M>(j, cl)
                        ? *reinterpret_cast<const v4bh*>(p.q + sbase + (size_t)j * rstride + vq)
                        : v4bh{0, 0, 0, 0};""",
    """                    v4bh qv = g2_row_ok<M>(j, cl)
                        ? *reinterpret_cast<const v4bh*>(p.q + sbase + (size_t)j * rstride + vq)
                        : v4bh{0, 0, 0, 0};
                    if (p.q_rstd != nullptr && g2_row_ok<M>(j, cl)) {   // kra F1
                        const float r_ = p.q_rstd[(size_t)(g2_row_tok<VL>(p.ctok, i) + j) * p.H + hh];
                        #pragma unroll
                        for (int e_ = 0; e_ < 4; ++e_) qv[e_] = bf_rne(bf2f(qv[e_]) * r_);
                    }""")
sub(wu, """extern "C" int g2_wu(const void* k, const void* v, const void* q, const void* beta,
                     const void* A, const void* g,
                     void* Wn, void* Wb, void* Un, void* KgA, void* Kn, void* Qn,
                     void* kgB, void* vB,
                     int B, int T, int H, const int* ctok, const int* clen, hipStream_t s) {""",
    """extern "C" int g2_wu_l2n(const void* k, const void* v, const void* q, const void* beta,
                         const void* A, const void* g,
                         void* Wn, void* Wb, void* Un, void* KgA, void* Kn, void* Qn,
                         void* kgB, void* vB,
                         int B, int T, int H, const int* ctok, const int* clen,
                         const void* q_rstd, const void* k_rstd, hipStream_t s) {""")
sub(wu, """                   H, T / 64, ctok, clen, stop};
    if (ctok)
        hipLaunchKernelGGL(g2::g2_wu_kernel<true>, dim3(T / 64, H), dim3(256), 0, s, p);
    else
        hipLaunchKernelGGL(g2::g2_wu_kernel<false>, dim3(T / 64, H), dim3(256), 0, s, p);
    return (int)hipGetLastError();
}""", """                   H, T / 64, ctok, clen, stop, (const float*)q_rstd, (const float*)k_rstd};
    if (ctok)
        hipLaunchKernelGGL(g2::g2_wu_kernel<true>, dim3(T / 64, H), dim3(256), 0, s, p);
    else
        hipLaunchKernelGGL(g2::g2_wu_kernel<false>, dim3(T / 64, H), dim3(256), 0, s, p);
    return (int)hipGetLastError();
}

// pre-F1 ABI (flash-train's runner): normalized q/k in, no rstd
extern "C" int g2_wu(const void* k, const void* v, const void* q, const void* beta,
                     const void* A, const void* g,
                     void* Wn, void* Wb, void* Un, void* KgA, void* Kn, void* Qn,
                     void* kgB, void* vB,
                     int B, int T, int H, const int* ctok, const int* clen, hipStream_t s) {
    return g2_wu_l2n(k, v, q, beta, A, g, Wn, Wb, Un, KgA, Kn, Qn, kgB, vB, B, T, H, ctok, clen,
                     nullptr, nullptr, s);
}""")

# ------------------------------------------------------------------ s4b
wy = G2 / "g2_wy.cuh"
sub(wy, """    const bf16* vnc;                    // kra P-22: C-form v_new (std order), read when v_new is null
};""", """    const bf16* vnc;                    // kra P-22: C-form v_new (std order), read when v_new is null
    const float *q_rstd, *k_rstd;       // kra F1: non-null -> q_row / k are RAW, l2norm on load
};""")
S4BT = "__device__ __forceinline__ void s4b_tail("
sub_in(wy, S4BT, """    const bf16 *q_ = p.q_row + cb, *k_ = p.k + cb;
    const float* g_ = p.g + cb;
    float *dq_ = p.dq + cb, *dk_ = p.dk + cb, *dg_ = p.dg + cb;
    const float* aq_ = p.dAqk + cbA;""", """    const bf16 *q_ = p.q_row + cb, *k_ = p.k + cb;
    const bool raw_ = p.k_rstd != nullptr;                         // kra F1
    const float qr_ = (raw_ && rok) ? p.q_rstd[(size_t)(tok0 + row) * H + hh] : 0.f;
    const float kr_ = (raw_ && rok) ? p.k_rstd[(size_t)(tok0 + row) * H + hh] : 0.f;
    const float* g_ = p.g + cb;
    float *dq_ = p.dq + cb, *dk_ = p.dk + cb, *dg_ = p.dg + cb;
    const float* aq_ = p.dAqk + cbA;""")
sub_in(wy, S4BT, """        const v4bh kk = rok ? ld8(k_ + ro + kc) : v4bh{0, 0, 0, 0};
        const v4bh qq = rok ? ld8(q_ + ro + kc) : v4bh{0, 0, 0, 0};""",
    """        v4bh kk = rok ? ld8(k_ + ro + kc) : v4bh{0, 0, 0, 0};
        v4bh qq = rok ? ld8(q_ + ro + kc) : v4bh{0, 0, 0, 0};
        if (raw_) {                                                 // kra F1: l2n_apply's value
            #pragma unroll
            for (int e_ = 0; e_ < 4; ++e_) {
                kk[e_] = bf_rne(bf2f(kk[e_]) * kr_);
                qq[e_] = bf_rne(bf2f(qq[e_]) * qr_);
            }
        }""")
sub(wy, """                                const int* clen, const void* do_row, const void* dv2c,
                                const void* vnc, hipStream_t s) {
    // T = chunk-padded rows (NT = T/64); with ctok/clen the row tensors are packed.""",
    """                                const int* clen, const void* do_row, const void* dv2c,
                                const void* vnc, const void* q_rstd, const void* k_rstd,
                                hipStream_t s) {
    // T = chunk-padded rows (NT = T/64); with ctok/clen the row tensors are packed.""")
sub(wy, """                       ctok, clen, (const bf16*)do_row, (const bf16*)vnc};
        if (ctok) hipLaunchKernelGGL(p3_wy_s4b_kernel<true>, dim3(NT, H), dim3(256), 0, s, pb);""",
    """                       ctok, clen, (const bf16*)do_row, (const bf16*)vnc,
                       (const float*)q_rstd, (const float*)k_rstd};
        if (ctok) hipLaunchKernelGGL(p3_wy_s4b_kernel<true>, dim3(NT, H), dim3(256), 0, s, pb);""")
# the split launcher was renamed in place: rename it _r2 and add the old _r as a forwarder
t = wy.read_text()
i = t.index("const void* vnc, const void* q_rstd, const void* k_rstd,")
j = t.rindex('extern "C" int g2_wy_s4_split_r(', 0, i)
t = t[:j] + 'extern "C" int g2_wy_s4_split_r2(' + t[j + len('extern "C" int g2_wy_s4_split_r('):]
fwd = '''
// pre-F1 ABI (flash-train's runner): normalized q_row/k in, no rstd
extern "C" int g2_wy_s4_split_r(const void* q_kgb, const void* k, const void* v_vb,
                                const void* v_new, const void* g, const void* beta,
                                const void* A, const void* h, const void* do_, const void* dh,
                                const void* dv, const void* dAqk, void* dq, void* dk,
                                void* dv2, void* dg, void* db, void* dA, void* dwT,
                                void* db_part, float scale, int T, int H, int NT,
                                const void* q_row, const void* v_row, const int* ctok,
                                const int* clen, const void* do_row, const void* dv2c,
                                const void* vnc, hipStream_t s) {
    return g2_wy_s4_split_r2(q_kgb, k, v_vb, v_new, g, beta, A, h, do_, dh, dv, dAqk, dq, dk, dv2,
                             dg, db, dA, dwT, db_part, scale, T, H, NT, q_row, v_row, ctok, clen,
                             do_row, dv2c, vnc, nullptr, nullptr, s);
}
'''
k = t.index('extern "C" int g2_wy_s4_split(const void* q_kgb')
t = t[:k] + fwd.lstrip("\n") + "\n" + t[k:]
wy.write_text(t)

# ------------------------------------------------------------------ bwd_tail (use_l2 == 2: raw q/k)
tl = G2 / "g2_bwd_tail.cuh"
sub(tl, """                            yv[w2][u] = *reinterpret_cast<const short2*>(y + so + u * (p.H * kK) + c2);
                            r4[w2][u] = rs[(size_t)(row + u) * p.H + hh];""",
    """                            yv[w2][u] = *reinterpret_cast<const short2*>(y + so + u * (p.H * kK) + c2);
                            r4[w2][u] = rs[(size_t)(row + u) * p.H + hh];
                            if (p.use_l2 == 2) {   // kra F1: y holds RAW q/k -> l2n_apply's value
                                yv[w2][u].x = bf_rne(bf2f(yv[w2][u].x) * r4[w2][u]);
                                yv[w2][u].y = bf_rne(bf2f(yv[w2][u].y) * r4[w2][u]);
                            }""")

# ------------------------------------------------------------------ bindings
bd = G2 / "g2_bindings.cpp"
sub(bd, """extern "C" int g2_wu(const void* k, const void* v, const void* q, const void* beta,
                     const void* A, const void* g,
                     void* Wn, void* Wb, void* Un, void* KgA, void* Kn, void* Qn,
                     void* kgB, void* vB,
                     int B, int T, int H, const int* ctok, const int* clen, hipStream_t s);""",
    """extern "C" int g2_wu_l2n(const void* k, const void* v, const void* q, const void* beta,
                         const void* A, const void* g,
                         void* Wn, void* Wb, void* Un, void* KgA, void* Kn, void* Qn,
                         void* kgB, void* vB,
                         int B, int T, int H, const int* ctok, const int* clen,
                         const void* q_rstd, const void* k_rstd, hipStream_t s);""")
sub(bd, """                                const int* clen, const void* do_row, const void* dv2c,
                                const void* vnc, hipStream_t s);
extern "C" int g2_wy_s4_split(""", """                                const int* clen, const void* do_row, const void* dv2c,
                                const void* vnc, hipStream_t s);
extern "C" int g2_wy_s4_split_r2(const void* q_kgb, const void* k, const void* v_vb,
                                 const void* v_new, const void* g, const void* beta,
                                 const void* A, const void* h, const void* do_, const void* dh,
                                 const void* dv, const void* dAqk, void* dq, void* dk,
                                 void* dv2, void* dg, void* db, void* dA, void* dwT,
                                 void* db_part, float scale, int T, int H, int NT,
                                 const void* q_row, const void* v_row, const int* ctok,
                                 const int* clen, const void* do_row, const void* dv2c,
                                 const void* vnc, const void* q_rstd, const void* k_rstd,
                                 hipStream_t s);
extern "C" int g2_wy_s4_split(""")
sub(bd, """                              c10::optional<torch::Tensor> clen, bool emit_p1, bool emit_vb) {""",
    """                              c10::optional<torch::Tensor> clen, bool emit_p1, bool emit_vb,
                              c10::optional<torch::Tensor> q_rstd,
                              c10::optional<torch::Tensor> k_rstd) {""")
sub(bd, """    const int rc = g2_wu(k.data_ptr(), v.data_ptr(), q.data_ptr(), beta.data_ptr(),""",
    """    // kra F1: rstd given -> k/q are RAW, the kernel applies bf16(x * rstd) on load
    const bool raw_qk = q_rstd.has_value() && q_rstd->defined() && k_rstd.has_value() && k_rstd->defined();
    if (raw_qk) { check_contig(*q_rstd, "wu q_rstd"); check_contig(*k_rstd, "wu k_rstd"); }
    const int rc = g2_wu_l2n(k.data_ptr(), v.data_ptr(), q.data_ptr(), beta.data_ptr(),""")
sub(bd, """                         1, (int)(chunks * 64), (int)heads, rm.ctok, rm.clen, cur_stream());
    TORCH_CHECK(rc == 0, "g2_wu failed: ", rc);""",
    """                         1, (int)(chunks * 64), (int)heads, rm.ctok, rm.clen,
                         raw_qk ? q_rstd->data_ptr() : nullptr, raw_qk ? k_rstd->data_ptr() : nullptr,
                         cur_stream());
    TORCH_CHECK(rc == 0, "g2_wu failed: ", rc);""")
sub(bd, """          py::arg("clen") = py::none(), py::arg("emit_p1") = true, py::arg("emit_vb") = true);""",
    """          py::arg("clen") = py::none(), py::arg("emit_p1") = true, py::arg("emit_vb") = true,
          py::arg("q_rstd") = py::none(), py::arg("k_rstd") = py::none());""")
sub(bd, """                                       c10::optional<torch::Tensor> vn_c,
                                       c10::optional<torch::Tensor> dv2_c) {""",
    """                                       c10::optional<torch::Tensor> vn_c,
                                       c10::optional<torch::Tensor> dv2_c,
                                       c10::optional<torch::Tensor> q_rstd,
                                       c10::optional<torch::Tensor> k_rstd) {
    // kra F1: rstd given -> k / q_row are RAW, s4b applies bf16(x * rstd) on load
    const bool raw_qk = q_rstd.has_value() && q_rstd->defined() && k_rstd.has_value() && k_rstd->defined();""")
sub(bd, """    const int rc = g2_wy_s4_split_r(q_kgb.data_ptr(), k.data_ptr(), has_vb ? v_vb.data_ptr() : nullptr,""",
    """    const int rc = g2_wy_s4_split_r2(q_kgb.data_ptr(), k.data_ptr(), has_vb ? v_vb.data_ptr() : nullptr,""")
sub(bd, """                                  has_vn ? nullptr : vn_c->data_ptr(), cur_stream());
    TORCH_CHECK(rc == 0, "g2_wy_s4_split launch failed: ", rc);""",
    """                                  has_vn ? nullptr : vn_c->data_ptr(),
                                  raw_qk ? q_rstd->data_ptr() : nullptr,
                                  raw_qk ? k_rstd->data_ptr() : nullptr, cur_stream());
    TORCH_CHECK(rc == 0, "g2_wy_s4_split launch failed: ", rc);""")
sub(bd, """                                    c10::optional<torch::Tensor> clen, bool gate_none) {""",
    """                                    c10::optional<torch::Tensor> clen, bool gate_none,
                                    bool qk_raw) {""")
sub(bd, """                               use_l2 ? 1 : 0, use_beta ? 1 : 0,""",
    """                               use_l2 ? (qk_raw ? 2 : 1) : 0, use_beta ? 1 : 0,   // kra F1: 2 = raw q/k""")

# ------------------------------------------------------------------ python wrappers + route
ops = ROOT / "hip_kda" / "ops" / "g2_ops.py"
sub(ops, """def wu(k, v, q, beta, A, g, heads, chunks, rowmap=None, emit_p1=True, emit_vb=True):""",
    """def wu(k, v, q, beta, A, g, heads, chunks, rowmap=None, emit_p1=True, emit_vb=True,
       q_rstd=None, k_rstd=None):""")
sub(ops, """                     *_rowmap(rowmap)[:2], bool(emit_p1), bool(emit_vb))""",
    """                     *_rowmap(rowmap)[:2], bool(emit_p1), bool(emit_vb),
                     None if q_rstd is None else q_rstd.contiguous(),
                     None if k_rstd is None else k_rstd.contiguous())""")
sub(ops, """             beta_raw, lower_bound, beta_scale, use_l2, use_beta, rowmap=None, gate_none=False):""",
    """             beta_raw, lower_bound, beta_scale, use_l2, use_beta, rowmap=None, gate_none=False,
             qk_raw=False):""")
sub(ops, """                           bool(use_l2), bool(use_beta), *_rowmap(rowmap)[:2], bool(gate_none))""",
    """                           bool(use_l2), bool(use_beta), *_rowmap(rowmap)[:2], bool(gate_none),
                           bool(qk_raw))""")
sub(ops, """                q_row, v_row, dAqk, heads, tokens, scale, emit_dA=False, rowmap=None, do_row=None,
                vn_c=None, dv2_c=None):""",
    """                q_row, v_row, dAqk, heads, tokens, scale, emit_dA=False, rowmap=None, do_row=None,
                vn_c=None, dv2_c=None, q_rstd=None, k_rstd=None):""")
sub(ops, """                              None if dv2_c is None else dv2_c.reshape(-1).contiguous())""",
    """                              None if dv2_c is None else dv2_c.reshape(-1).contiguous(),
                              None if q_rstd is None else q_rstd.contiguous(),
                              None if k_rstd is None else k_rstd.contiguous())""")

k3 = ROOT / "hip_kda" / "ops" / "k3_g2.py"
sub(k3, """        qn, kn = q, k
        if ctx.use_qk_l2norm_in_kernel:
            if os.environ.get("KDA_G2_BL2", "1").strip() != "0":""",
    """        qn, kn = q, k
        # kra F1: on the production S4 split route the consumers (wu, s4b, bwd_tail) apply the
        # saved rstd on load -- bf16(x * rstd), l2n_apply's exact value -- so qn/kn are never
        # written (KDA_G2_F1=0 restores l2n_apply).
        raw_qk = (ctx.use_qk_l2norm_in_kernel and hipi and q_rstd is not None and k_rstd is not None
                  and os.environ.get("KDA_G2_S4", "1").strip() != "0"
                  and os.environ.get("KDA_G2_P4", "3").strip() == "3"
                  and os.environ.get("KDA_G2_F1", "1").strip() != "0")
        if ctx.use_qk_l2norm_in_kernel and not raw_qk:
            if os.environ.get("KDA_G2_BL2", "1").strip() != "0":""")
sub(k3, """        Wn, Wb, Un, KgA, Kn, Qn, kgB, vB = g2_ops.wu(kn, v, qn, beta, Akk, g, H, NT, rowmap=lay,
                                                     emit_p1=not kept, emit_vb=not no_vb)""",
    """        Wn, Wb, Un, KgA, Kn, Qn, kgB, vB = g2_ops.wu(kn, v, qn, beta, Akk, g, H, NT, rowmap=lay,
                                                     emit_p1=not kept, emit_vb=not no_vb,
                                                     q_rstd=q_rstd if raw_qk else None,
                                                     k_rstd=k_rstd if raw_qk else None)""")
sub(k3, """                                                             do_row=do if no_ando else None,
                                                             vn_c=vn_c, dv2_c=dv2_c)""",
    """                                                             do_row=do if no_ando else None,
                                                             vn_c=vn_c, dv2_c=dv2_c,
                                                             q_rstd=q_rstd if raw_qk else None,
                                                             k_rstd=k_rstd if raw_qk else None)""")
sub(k3, """                ctx.use_qk_l2norm_in_kernel, ctx.use_beta_sigmoid_in_kernel, rowmap=lay,
                gate_none=pg)""",
    """                ctx.use_qk_l2norm_in_kernel, ctx.use_beta_sigmoid_in_kernel, rowmap=lay,
                gate_none=pg, qk_raw=raw_qk)""")
print("P-36 (F1) APPLIED")
