"""kra P-20 (global route, round 4, structural): drop the aN_do layout copy.  aN_do (A form,
P3 order) is dav's scattered 8 B emit of do (ablation on the current baseline: ~660 us of dav's
1976 us at 8K/96) and its ONLY consumer is s4b's prologue (8 x 8 B per lane, once per CTA).  In
that layout s4b's fragment (A*8 + vt, lane l) is do[16A + (l&15)][16vt + 4(l>>4) .. +3], i.e.
the row-major quad at (row, 16vt + 4q): s4b now loads it from the row-major do (same row mask
the dav emit used -> varlen padding reads as 0), and dav skips the emit.  New C entry
g2_wy_s4_split_r (+ do_row); g2_wy_s4_split keeps its signature (flash-train).  Production
route only (S4=1, P4=3; KDA_G2_NOADO=0 restores the copy).  Values unchanged -> bit-identical.
Usage: python3 p20_s4b_do_row.py <tree root>"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])


def sub(path, old, new, count=1):
    t = path.read_text()
    assert t.count(old) == count, (path.name, old[:90], t.count(old))
    path.write_text(t.replace(old, new))


WY = ROOT / "csrc/g2/g2_wy.cuh"
DAV = ROOT / "csrc/g2/g2_dav.cuh"
BI = ROOT / "csrc/g2/g2_bindings.cpp"
OPS = ROOT / "hip_kda/ops/g2_ops.py"
K3 = ROOT / "hip_kda/ops/k3_g2.py"

# ---------------------------------------------------------------- s4b
sub(WY, """struct WyS4BParams {
    const bf16 *q_row, *k, *A, *h, *do_, *dh, *v_new;
    const float *g, *beta, *dAqk;
    const bf16* dwT;
    const float *dA, *db_part;
    float *dq, *dk, *dg, *db;
    float scale;
    int T, H, NT;
    const int* ctok; const int* clen;   // varlen phase 2 (null = dense)
};""", """struct WyS4BParams {
    const bf16 *q_row, *k, *A, *h, *do_, *dh, *v_new;
    const float *g, *beta, *dAqk;
    const bf16* dwT;
    const float *dA, *db_part;
    float *dq, *dk, *dg, *db;
    float scale;
    int T, H, NT;
    const int* ctok; const int* clen;   // varlen phase 2 (null = dense)
    const bf16* do_row;                 // kra P-20: row-major do, read when do_ (aN_do) is null
};""")
sub(WY, """    v4bh ado[8], avn[8];
    {
        const bf16* doN = p.do_ + g2_nat_blk(i, hh, H, 0, true, 4 * 8 * 256);
        #pragma unroll
        for (int vt = 0; vt < 8; ++vt) ado[vt] = ld8(doN + ((A * 8 + vt) * 64 + l) * 4);
        const bf16* vnN = p.v_new + g2_nat_blk(i, hh, H, 0, true, 4 * 8 * 256);""",
    """    v4bh ado[8], avn[8];
    {
        if (p.do_ != nullptr) {
            const bf16* doN = p.do_ + g2_nat_blk(i, hh, H, 0, true, 4 * 8 * 256);
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) ado[vt] = ld8(doN + ((A * 8 + vt) * 64 + l) * 4);
        } else {
            // kra P-20: the aN_do fragment (A*8 + vt, l) = do[row][16vt + 4q..+3], row = 16A + m
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt)
                ado[vt] = rok ? ld8(p.do_row + cb + ro + 16 * vt) : v4bh{0, 0, 0, 0};
        }
        const bf16* vnN = p.v_new + g2_nat_blk(i, hh, H, 0, true, 4 * 8 * 256);""", count=1)
sub(WY, """extern "C" int g2_wy_s4_split(const void* q_kgb, const void* k, const void* v_vb,
                              const void* v_new, const void* g, const void* beta,
                              const void* A, const void* h, const void* do_, const void* dh,
                              const void* dv, const void* dAqk, void* dq, void* dk,
                              void* dv2, void* dg, void* db, void* dA, void* dwT,
                              void* db_part, float scale, int T, int H, int NT,
                              const void* q_row, const void* v_row, const int* ctok,
                              const int* clen, hipStream_t s) {""",
    """// kra P-20: + do_row (row-major do; read by s4b when do_ == nullptr)
extern "C" int g2_wy_s4_split_r(const void* q_kgb, const void* k, const void* v_vb,
                                const void* v_new, const void* g, const void* beta,
                                const void* A, const void* h, const void* do_, const void* dh,
                                const void* dv, const void* dAqk, void* dq, void* dk,
                                void* dv2, void* dg, void* db, void* dA, void* dwT,
                                void* db_part, float scale, int T, int H, int NT,
                                const void* q_row, const void* v_row, const int* ctok,
                                const int* clen, const void* do_row, hipStream_t s) {""")
sub(WY, """                       (float*)dq, (float*)dk, (float*)dg, (float*)db, scale, T, H, NT,
                       ctok, clen};
        if (ctok) hipLaunchKernelGGL(p3_wy_s4b_kernel<true>, dim3(NT, H), dim3(256), 0, s, pb);
        else hipLaunchKernelGGL(p3_wy_s4b_kernel<false>, dim3(NT, H), dim3(256), 0, s, pb);
        return (int)hipGetLastError();
    }
}""", """                       (float*)dq, (float*)dk, (float*)dg, (float*)db, scale, T, H, NT,
                       ctok, clen, (const bf16*)do_row};
        if (ctok) hipLaunchKernelGGL(p3_wy_s4b_kernel<true>, dim3(NT, H), dim3(256), 0, s, pb);
        else hipLaunchKernelGGL(p3_wy_s4b_kernel<false>, dim3(NT, H), dim3(256), 0, s, pb);
        return (int)hipGetLastError();
    }
}

extern "C" int g2_wy_s4_split(const void* q_kgb, const void* k, const void* v_vb,
                              const void* v_new, const void* g, const void* beta,
                              const void* A, const void* h, const void* do_, const void* dh,
                              const void* dv, const void* dAqk, void* dq, void* dk,
                              void* dv2, void* dg, void* db, void* dA, void* dwT,
                              void* db_part, float scale, int T, int H, int NT,
                              const void* q_row, const void* v_row, const int* ctok,
                              const int* clen, hipStream_t s) {
    return g2_wy_s4_split_r(q_kgb, k, v_vb, v_new, g, beta, A, h, do_, dh, dv, dAqk, dq, dk, dv2,
                            dg, db, dA, dwT, db_part, scale, T, H, NT, q_row, v_row, ctok, clen,
                            nullptr, s);
}""")

# ---------------------------------------------------------------- dav: emit optional
sub(DAV, """            *reinterpret_cast<v4bh*>(p.aNdo + nblk3 + (size_t)(4 * (j & 15)
                + 64 * ((vq & 15) >> 2) + 256 * (vq >> 4) + 2048 * (j >> 4))) = dv_;""",
    """            if (p.aNdo != nullptr)   // kra P-20: production route -- s4b reads do row-major
                *reinterpret_cast<v4bh*>(p.aNdo + nblk3 + (size_t)(4 * (j & 15)
                    + 64 * ((vq & 15) >> 2) + 256 * (vq >> 4) + 2048 * (j >> 4))) = dv_;""")

# ---------------------------------------------------------------- bindings
sub(BI, """extern "C" int g2_wy_s4_split(const void* q_kgb, const void* k, const void* v_vb,""",
    """extern "C" int g2_wy_s4_split_r(const void* q_kgb, const void* k, const void* v_vb,
                                const void* v_new, const void* g, const void* beta,
                                const void* A, const void* h, const void* do_, const void* dh,
                                const void* dv, const void* dAqk, void* dq, void* dk,
                                void* dv2, void* dg, void* db, void* dA, void* dwT,
                                void* db_part, float scale, int T, int H, int NT,
                                const void* q_row, const void* v_row, const int* ctok,
                                const int* clen, const void* do_row, hipStream_t s);
extern "C" int g2_wy_s4_split(const void* q_kgb, const void* k, const void* v_vb,""")
sub(BI, """std::vector<torch::Tensor> dav(torch::Tensor do_, torch::Tensor Vn, torch::Tensor A,
                               double scale, int64_t heads, int64_t chunks,
                               c10::optional<torch::Tensor> ctok,
                               c10::optional<torch::Tensor> clen) {""",
    """std::vector<torch::Tensor> dav(torch::Tensor do_, torch::Tensor Vn, torch::Tensor A,
                               double scale, int64_t heads, int64_t chunks,
                               c10::optional<torch::Tensor> ctok,
                               c10::optional<torch::Tensor> clen, bool emit_ando) {""")
sub(BI, """    auto aNdo = torch::empty({1, chunks, heads, 4, 8, 4, 16, 4}, do_.options());
    const int rc = g2_dav(do_.data_ptr(), Vn.data_ptr(), A.data_ptr(), dA.data_ptr(),
                          dVn.data_ptr(), dOT.data_ptr(), aNdo.data_ptr(),""",
    """    // kra P-20: emit_ando=false (production S4 split route): s4b reads do row-major
    auto aNdo = emit_ando ? torch::empty({1, chunks, heads, 4, 8, 4, 16, 4}, do_.options())
                          : torch::Tensor();
    const int rc = g2_dav(do_.data_ptr(), Vn.data_ptr(), A.data_ptr(), dA.data_ptr(),
                          dVn.data_ptr(), dOT.data_ptr(), emit_ando ? aNdo.data_ptr() : nullptr,""")
sub(BI, """    m.def("dav", &dav, "G2 HIP chunk_kda_bwd_dAv, native v_new in / dVn out (S3-11)");""",
    """    m.def("dav", &dav, "G2 HIP chunk_kda_bwd_dAv, native v_new in / dVn out (S3-11)",
          py::arg("do_"), py::arg("Vn"), py::arg("A"), py::arg("scale"), py::arg("heads"),
          py::arg("chunks"), py::arg("ctok") = py::none(), py::arg("clen") = py::none(),
          py::arg("emit_ando") = true);""")
sub(BI, """                                       c10::optional<torch::Tensor> ctok,
                                       c10::optional<torch::Tensor> clen) {
    // kra P-17: an EMPTY v_vb selects the no-copy route (s4a stages the row-major v on chip)
    const bool has_vb = v_vb.numel() > 0;
    for (auto* t : {&q_kgb, &k, &v_new, &A, &h, &do_, &dh, &dv, &q_row, &v_row}) {
        check_contig(*t, "wy_s4_split input"); check_bf16(*t, "wy_s4_split input");
    }""",
    """                                       c10::optional<torch::Tensor> ctok,
                                       c10::optional<torch::Tensor> clen,
                                       c10::optional<torch::Tensor> do_row) {
    // kra P-17: an EMPTY v_vb selects the no-copy route (s4a stages the row-major v on chip)
    const bool has_vb = v_vb.numel() > 0;
    // kra P-20: an EMPTY do_ (aN_do) selects the row-major do route (s4b prologue)
    const bool has_ando = do_.numel() > 0;
    TORCH_CHECK(has_ando || (do_row.has_value() && do_row->numel() > 0),
                "g2: wy_s4_split needs aN_do or do_row");
    for (auto* t : {&q_kgb, &k, &v_new, &A, &h, &dh, &dv, &q_row, &v_row}) {
        check_contig(*t, "wy_s4_split input"); check_bf16(*t, "wy_s4_split input");
    }
    if (has_ando) { check_contig(do_, "wy_s4_split do_"); check_bf16(do_, "wy_s4_split do_"); }
    else { check_contig(*do_row, "wy_s4_split do_row"); check_bf16(*do_row, "wy_s4_split do_row"); }""")
sub(BI, """                want_nat(do_) && want_nat(dv) && want_nat(q_kgb) && (!has_vb || want_nat(v_vb)) &&""",
    """                (has_ando ? want_nat(do_) : want_row(*do_row)) && want_nat(dv) && want_nat(q_kgb) &&
                (!has_vb || want_nat(v_vb)) &&""")
sub(BI, """    const int rc = g2_wy_s4_split(q_kgb.data_ptr(), k.data_ptr(), has_vb ? v_vb.data_ptr() : nullptr,
                                  v_new.data_ptr(),
                                  g.data_ptr(), beta.data_ptr(), A.data_ptr(), h.data_ptr(),
                                  do_.data_ptr(), dh.data_ptr(), dv.data_ptr(), dAqk.data_ptr(),""",
    """    const int rc = g2_wy_s4_split_r(q_kgb.data_ptr(), k.data_ptr(), has_vb ? v_vb.data_ptr() : nullptr,
                                  v_new.data_ptr(),
                                  g.data_ptr(), beta.data_ptr(), A.data_ptr(), h.data_ptr(),
                                  has_ando ? do_.data_ptr() : nullptr, dh.data_ptr(), dv.data_ptr(),
                                  dAqk.data_ptr(),""")
sub(BI, """                                  q_row.data_ptr(), v_row.data_ptr(), rm.ctok, rm.clen,
                                  cur_stream());
    TORCH_CHECK(rc == 0, "g2_wy_s4_split launch failed: ", rc);""",
    """                                  q_row.data_ptr(), v_row.data_ptr(), rm.ctok, rm.clen,
                                  has_ando ? nullptr : do_row->data_ptr(), cur_stream());
    TORCH_CHECK(rc == 0, "g2_wy_s4_split launch failed: ", rc);""")

# ---------------------------------------------------------------- python
sub(OPS, """def dav(do, Vn, A, scale, heads, chunks, rowmap=None):""",
    """def dav(do, Vn, A, scale, heads, chunks, rowmap=None, emit_ando=True):""")
sub(OPS, """    return _ext().dav(do.contiguous(), Vn.contiguous(), A.contiguous(), float(scale),
                      int(heads), int(chunks), *_rowmap(rowmap)[:2])""",
    """    return _ext().dav(do.contiguous(), Vn.contiguous(), A.contiguous(), float(scale),
                      int(heads), int(chunks), *_rowmap(rowmap)[:2], bool(emit_ando))""")
sub(OPS, """def wy_s4_split(q_kgb, k_row, v_vb, v_new_n, g, beta, a, h_n, do_n, dh_n, dv_n,
                q_row, v_row, dAqk, heads, tokens, scale, emit_dA=False, rowmap=None):""",
    """def wy_s4_split(q_kgb, k_row, v_vb, v_new_n, g, beta, a, h_n, do_n, dh_n, dv_n,
                q_row, v_row, dAqk, heads, tokens, scale, emit_dA=False, rowmap=None, do_row=None):""")
sub(OPS, """                              float(scale), bool(emit_dA), *_rowmap(rowmap)[:2])""",
    """                              float(scale), bool(emit_dA), *_rowmap(rowmap)[:2],
                              None if do_row is None else do_row.reshape(-1).contiguous())""")
sub(K3, """        dAqk, dVn, dOT, aN_do = g2_ops.dav(do, Vn, Aqk, scale, H, NT, rowmap=lay)""",
    """        # kra P-20: on the S4 split route s4b reads do row-major; dav skips the aN_do copy
        no_ando = no_vb and os.environ.get("KDA_G2_NOADO", "1").strip() != "0"
        dAqk, dVn, dOT, aN_do = g2_ops.dav(do, Vn, Aqk, scale, H, NT, rowmap=lay,
                                           emit_ando=not no_ando)
        if no_ando:
            aN_do = torch.empty(0, dtype=do.dtype, device=do.device)""")
sub(K3, """                dq, dk, dv_grad, dg, db = g2_ops.wy_s4_split(*wy_args, dAqk, H, 64 * NT, scale,
                                                             rowmap=lay)""",
    """                dq, dk, dv_grad, dg, db = g2_ops.wy_s4_split(*wy_args, dAqk, H, 64 * NT, scale,
                                                             rowmap=lay,
                                                             do_row=do if no_ando else None)""")
print("P-20 APPLIED")
