"""kra P-22 (global route, round 5, structural): the two backward recurrences stop emitting their
aN copies.  Every chunk step, fwd_h_bn (v8o4) wrote aN(v_new) and dhu (v8b) wrote aN(dv2) as 8
scattered 2 B stores per lane -- in the VMEM-issue-bound, chain-latency-bound recurrences (8K/12:
60% CUs busy, chain 15-18%).  Their only consumers are the s4b / s4a prologues (once per CTA).
The C-form copies hold the SAME values: v8o4 already writes Vn; dhu now writes dV2n (2 x 8 B
contiguous per step) in a new compile-time MODE 2.  The consumers rebuild the A-form fragment with
a 4x4 transpose across the 4 lanes sharing a row (two 32-bit shfl_xor: l^32, l^16; offline check:
0/256 mismatches) -- no LDS, no barrier.  New compile-time v8o4 MODE 4 (HbN + Vn).  Keep-state
keeps the forward's saved aN_vn.  KDA_G2_NOAN=0 restores the copies.  Bit-identical.
Usage: python3 p22_no_an_copies.py <tree root>          (on top of P-20)"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])


def sub(path, old, new, count=1):
    t = path.read_text()
    assert t.count(old) == count, (path.name, old[:90], t.count(old))
    path.write_text(t.replace(old, new))


V8 = ROOT / "csrc/g2/g2_fwd_v8.cuh"
EN = ROOT / "csrc/g2/g2_engine.cuh"
WY = ROOT / "csrc/g2/g2_wy.cuh"
BI = ROOT / "csrc/g2/g2_bindings.cpp"
OPS = ROOT / "hip_kda/ops/g2_ops.py"
K3 = ROOT / "hip_kda/ops/k3_g2.py"

# ---------------------------------------------------------------- v8o4 MODE 4
sub(V8, """    // MODE 0 generic | 1 fwd (Ob) | 2 fwd keep-state (Ob+HbN+Vn+ANv) | 3 bwd recompute (HbN+Vn+ANv)
    const bool hasO  = MODE == 0 ? p.O != nullptr : false;
    const bool hasOb = MODE == 0 ? p.Ob != nullptr : MODE != 3;
    const bool hasBN = MODE == 0 ? p.HbN != nullptr : MODE != 1;
    const bool hasVn = MODE == 0 ? p.Vn != nullptr : MODE != 1;
    const bool hasAN = MODE == 0 ? p.ANv != nullptr : MODE != 1;""",
    """    // MODE 0 generic | 1 fwd (Ob) | 2 fwd keep-state (Ob+HbN+Vn+ANv) | 3 bwd recompute (HbN+Vn+ANv)
    // | 4 bwd recompute without the aN copy (HbN+Vn; s4b rebuilds aN from Vn) [kra P-22]
    const bool hasO  = MODE == 0 ? p.O != nullptr : false;
    const bool hasOb = MODE == 0 ? p.Ob != nullptr : (MODE != 3 && MODE != 4);
    const bool hasBN = MODE == 0 ? p.HbN != nullptr : MODE != 1;
    const bool hasVn = MODE == 0 ? p.Vn != nullptr : MODE != 1;
    const bool hasAN = MODE == 0 ? p.ANv != nullptr : (MODE != 1 && MODE != 4);""")
sub(V8, """    else if (O == nullptr && Ob == nullptr && Vn && HbN && ANv) mode = 3;""",
    """    else if (O == nullptr && Ob == nullptr && Vn && HbN && ANv) mode = 3;
    else if (O == nullptr && Ob == nullptr && Vn && HbN && !ANv) mode = 4;   // kra P-22""")
sub(V8, """            case 3: hipLaunchKernelGGL((g2_fwd_h_v8o4_kernel<VL, 3>), grid, dim3(256), 0, s, p); break;""",
    """            case 3: hipLaunchKernelGGL((g2_fwd_h_v8o4_kernel<VL, 3>), grid, dim3(256), 0, s, p); break;
            case 4: hipLaunchKernelGGL((g2_fwd_h_v8o4_kernel<VL, 4>), grid, dim3(256), 0, s, p); break;""")

# ---------------------------------------------------------------- dhu v8b MODE 2
sub(EN, """    const bool hasDV2 = MODE == 0 ? p.dv2 != nullptr : false;
    const bool hasAN2 = MODE == 0 ? p.ANv2 != nullptr : true;""",
    """    const bool hasDV2 = MODE == 0 ? p.dv2 != nullptr : MODE == 2;    // kra P-22: MODE 2 = HbN + dv2
    const bool hasAN2 = MODE == 0 ? p.ANv2 != nullptr : MODE == 1;""", count=2)
sub(EN, """    const bool prod = HbN != nullptr && ANv2 != nullptr && dv2 == nullptr;""",
    """    const bool prod = HbN != nullptr && ANv2 != nullptr && dv2 == nullptr;
    const bool prod2 = HbN != nullptr && ANv2 == nullptr && dv2 != nullptr;   // kra P-22""")
sub(EN, """        if (prod) hipLaunchKernelGGL(g2_dhu_v8b_kernel<1>, dim3(kK / kBV, nseg * H), dim3(512), 0, s, p);
        else      hipLaunchKernelGGL(g2_dhu_v8b_kernel<0>, dim3(kK / kBV, nseg * H), dim3(512), 0, s, p);""",
    """        if (prod) hipLaunchKernelGGL(g2_dhu_v8b_kernel<1>, dim3(kK / kBV, nseg * H), dim3(512), 0, s, p);
        else if (prod2) hipLaunchKernelGGL(g2_dhu_v8b_kernel<2>, dim3(kK / kBV, nseg * H), dim3(512), 0, s, p);
        else      hipLaunchKernelGGL(g2_dhu_v8b_kernel<0>, dim3(kK / kBV, nseg * H), dim3(512), 0, s, p);""")

# ---------------------------------------------------------------- s4a / s4b
sub(WY, """struct WyS4AParams {""",
    """// kra P-22: C-form fragment (lane l: row l&15, cols (l>>4) + 4e) -> A-form fragment (row l&15,
// cols 4(l>>4) + e): a 4x4 transpose across the 4 lanes sharing the row, in two 32-bit exchanges
// (swap q bit 1 <-> e bit 1 with lane l^32, then q bit 0 <-> e bit 0 with lane l^16).
__device__ __forceinline__ v4bh g2_c2a(v4bh c, int l) {
    uint32_t d0 = (uint32_t)(unsigned short)c[0] | ((uint32_t)(unsigned short)c[1] << 16);
    uint32_t d1 = (uint32_t)(unsigned short)c[2] | ((uint32_t)(unsigned short)c[3] << 16);
    const bool b1 = (l >> 5) & 1, b0 = (l >> 4) & 1;
    uint32_t r = (uint32_t)__shfl_xor((int)(b1 ? d0 : d1), 32, 64);
    if (b1) d0 = r; else d1 = r;
    const uint32_t snd = b0 ? ((d0 & 0xFFFFu) | (d1 << 16)) : ((d0 >> 16) | (d1 & 0xFFFF0000u));
    r = (uint32_t)__shfl_xor((int)snd, 16, 64);
    if (b0) { d0 = (r & 0xFFFFu) | (d0 & 0xFFFF0000u); d1 = (r >> 16) | (d1 & 0xFFFF0000u); }
    else    { d0 = (d0 & 0xFFFFu) | (r << 16);          d1 = (d1 & 0xFFFFu) | (r & 0xFFFF0000u); }
    return v4bh{(short)(d0 & 0xFFFFu), (short)(d0 >> 16), (short)(d1 & 0xFFFFu), (short)(d1 >> 16)};
}

struct WyS4AParams {""")
sub(WY, """    float *dA, *db_part;
    int T, H, NT;
    const int* ctok; const int* clen;   // varlen phase 2 (null = dense)
};

struct WyS4BParams {""", """    float *dA, *db_part;
    int T, H, NT;
    const int* ctok; const int* clen;   // varlen phase 2 (null = dense)
    const bf16* dv2c;                   // kra P-22: C-form dv2 (std order), read when dv is null
};

struct WyS4BParams {""")
sub(WY, """        v4bh dva[8];
        #pragma unroll
        for (int vt = 0; vt < 8; ++vt) dva[vt] = aN(dvN, vt);
        #pragma unroll""",
    """        v4bh dva[8];
        if (p.dv != nullptr) {
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) dva[vt] = aN(dvN, vt);
        } else {                // kra P-22: from the C-form dV2n via the in-register transpose
            const bf16* dvc = p.dv2c + g2_nat_blk(i, hh, H, p.NT, false, 8192) + r * 8 * 256 + l * 4;
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) dva[vt] = g2_c2a(ld8(dvc + vt * 256), l);
        }
        #pragma unroll""")
sub(WY, """    const int* ctok; const int* clen;   // varlen phase 2 (null = dense)
    const bf16* do_row;                 // kra P-20: row-major do, read when do_ (aN_do) is null
};""", """    const int* ctok; const int* clen;   // varlen phase 2 (null = dense)
    const bf16* do_row;                 // kra P-20: row-major do, read when do_ (aN_do) is null
    const bf16* vnc;                    // kra P-22: C-form v_new (std order), read when v_new is null
};""")
sub(WY, """        const bf16* vnN = p.v_new + g2_nat_blk(i, hh, H, 0, true, 4 * 8 * 256);
        #pragma unroll
        for (int vt = 0; vt < 8; ++vt) avn[vt] = ld8(vnN + ((A * 8 + vt) * 64 + l) * 4);""",
    """        if (p.v_new != nullptr) {
            const bf16* vnN = p.v_new + g2_nat_blk(i, hh, H, 0, true, 4 * 8 * 256);
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) avn[vt] = ld8(vnN + ((A * 8 + vt) * 64 + l) * 4);
        } else {                // kra P-22: from the C-form Vn via the in-register transpose
            const bf16* vnc = p.vnc + g2_nat_blk(i, hh, H, p.NT, false, 8192) + A * 8 * 256 + l * 4;
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) avn[vt] = g2_c2a(ld8(vnc + vt * 256), l);
        }""")
sub(WY, """                                const int* clen, const void* do_row, hipStream_t s) {""",
    """                                const int* clen, const void* do_row, const void* dv2c,
                                const void* vnc, hipStream_t s) {""")
sub(WY, """                       (bf16*)dv2, (bf16*)dwT, (float*)dA, (float*)db_part, T, H, NT,
                       ctok, clen};""",
    """                       (bf16*)dv2, (bf16*)dwT, (float*)dA, (float*)db_part, T, H, NT,
                       ctok, clen, (const bf16*)dv2c};""")
sub(WY, """                       ctok, clen, (const bf16*)do_row};""",
    """                       ctok, clen, (const bf16*)do_row, (const bf16*)vnc};""")
sub(WY, """                            dg, db, dA, dwT, db_part, scale, T, H, NT, q_row, v_row, ctok, clen,
                            nullptr, s);""",
    """                            dg, db, dA, dwT, db_part, scale, T, H, NT, q_row, v_row, ctok, clen,
                            nullptr, nullptr, nullptr, s);""")

# ---------------------------------------------------------------- bindings
sub(BI, """                                const int* clen, const void* do_row, hipStream_t s);""",
    """                                const int* clen, const void* do_row, const void* dv2c,
                                const void* vnc, hipStream_t s);""")
sub(BI, """std::vector<torch::Tensor> fwd_h_bn(torch::Tensor Wn, torch::Tensor Kn, torch::Tensor Un,
                                    torch::Tensor Gn, c10::optional<torch::Tensor> seg) {""",
    """std::vector<torch::Tensor> fwd_h_bn(torch::Tensor Wn, torch::Tensor Kn, torch::Tensor Un,
                                    torch::Tensor Gn, c10::optional<torch::Tensor> seg,
                                    bool emit_an) {""")
sub(BI, """    auto ANv = torch::empty({B, NT, H, 4, 8, 4, 16, 4}, Wn.options());
    const auto sgb = seg_table(seg, Wn);
    const int rc = g2_fwd_h_v8o(Wn.data_ptr(), Kn.data_ptr(), Un.data_ptr(), Gn.data_ptr(),
                                nullptr, nullptr, nullptr, nullptr, Vn.data_ptr(),
                                HbN.data_ptr(), ANv.data_ptr(), 1.f, (int)B, (int)T,""",
    """    // kra P-22: emit_an=false -> no aN(v_new) copy (s4b rebuilds it from Vn)
    auto ANv = emit_an ? torch::empty({B, NT, H, 4, 8, 4, 16, 4}, Wn.options()) : torch::Tensor();
    const auto sgb = seg_table(seg, Wn);
    const int rc = g2_fwd_h_v8o(Wn.data_ptr(), Kn.data_ptr(), Un.data_ptr(), Gn.data_ptr(),
                                nullptr, nullptr, nullptr, nullptr, Vn.data_ptr(),
                                HbN.data_ptr(), emit_an ? ANv.data_ptr() : nullptr, 1.f, (int)B, (int)T,""")
sub(BI, """                                  double scale, bool emit_dv2n,
                                  c10::optional<torch::Tensor> seg) {""",
    """                                  double scale, bool emit_dv2n,
                                  c10::optional<torch::Tensor> seg, bool emit_an2) {""")
sub(BI, """    auto ANv2 = torch::empty({B, NT, H, 4, 8, 4, 16, 4}, KgA.options());
    const auto sgd = seg_table(seg, KgA);""",
    """    // kra P-22: emit_an2=false -> no aN(dv2) copy (s4a rebuilds it from the C-form dV2n)
    auto ANv2 = emit_an2 ? torch::empty({B, NT, H, 4, 8, 4, 16, 4}, KgA.options()) : torch::Tensor();
    const auto sgd = seg_table(seg, KgA);""")
sub(BI, """                              emit_dv2n ? dV2n.data_ptr() : nullptr, HbN.data_ptr(),
                              ANv2.data_ptr(), (float)scale, (int)B, (int)T, (int)H,""",
    """                              emit_dv2n ? dV2n.data_ptr() : nullptr, HbN.data_ptr(),
                              emit_an2 ? ANv2.data_ptr() : nullptr, (float)scale, (int)B, (int)T, (int)H,""")
sub(BI, """         py::arg("Wn"), py::arg("Kn"), py::arg("Un"), py::arg("Gn"),
         py::arg("seg") = py::none());
    m.def("dhu_bn", &dhu_bn, "G2 P2 emitting bN(dh)",""",
    """         py::arg("Wn"), py::arg("Kn"), py::arg("Un"), py::arg("Gn"),
         py::arg("seg") = py::none(), py::arg("emit_an") = true);
    m.def("dhu_bn", &dhu_bn, "G2 P2 emitting bN(dh)",""")
t = BI.read_text()
i = t.index('    m.def("dhu_bn", &dhu_bn, "G2 P2 emitting bN(dh)",')
j = t.index(");", i)
seg_part = t[i:j]
assert 'py::arg("seg")' in seg_part, seg_part
BI.write_text(t[:j] + ', py::arg("emit_an2") = true' + t[j:])
sub(BI, """                                       c10::optional<torch::Tensor> do_row) {""",
    """                                       c10::optional<torch::Tensor> do_row,
                                       c10::optional<torch::Tensor> vn_c,
                                       c10::optional<torch::Tensor> dv2_c) {
    // kra P-22: EMPTY v_new (aN_vn) / dv (aN_dv2) -> s4b / s4a rebuild them from the C-form copies
    const bool has_vn = v_new.numel() > 0, has_dv = dv.numel() > 0;
    TORCH_CHECK(has_vn || (vn_c.has_value() && vn_c->numel() > 0), "g2: wy_s4_split needs aN_vn or vn_c");
    TORCH_CHECK(has_dv || (dv2_c.has_value() && dv2_c->numel() > 0), "g2: wy_s4_split needs aN_dv2 or dv2_c");""")
sub(BI, """    for (auto* t : {&q_kgb, &k, &v_new, &A, &h, &dh, &dv, &q_row, &v_row}) {
        check_contig(*t, "wy_s4_split input"); check_bf16(*t, "wy_s4_split input");
    }""",
    """    for (auto* t : {&q_kgb, &k, &A, &h, &dh, &q_row, &v_row}) {
        check_contig(*t, "wy_s4_split input"); check_bf16(*t, "wy_s4_split input");
    }
    const torch::Tensor& vn_t = has_vn ? v_new : *vn_c;
    const torch::Tensor& dv_t = has_dv ? dv : *dv2_c;
    check_contig(vn_t, "wy_s4_split v_new"); check_bf16(vn_t, "wy_s4_split v_new");
    check_contig(dv_t, "wy_s4_split dv"); check_bf16(dv_t, "wy_s4_split dv");""")
sub(BI, """    TORCH_CHECK(want_row(k) && want_nat(v_new) && want_row(q_row) && want_row(v_row) &&""",
    """    TORCH_CHECK(want_row(k) && want_nat(vn_t) && want_row(q_row) && want_row(v_row) &&""")
sub(BI, """                (has_ando ? want_nat(do_) : want_row(*do_row)) && want_nat(dv) && want_nat(q_kgb) &&""",
    """                (has_ando ? want_nat(do_) : want_row(*do_row)) && want_nat(dv_t) && want_nat(q_kgb) &&""")
sub(BI, """    const int rc = g2_wy_s4_split_r(q_kgb.data_ptr(), k.data_ptr(), has_vb ? v_vb.data_ptr() : nullptr,
                                  v_new.data_ptr(),
                                  g.data_ptr(), beta.data_ptr(), A.data_ptr(), h.data_ptr(),
                                  has_ando ? do_.data_ptr() : nullptr, dh.data_ptr(), dv.data_ptr(),
                                  dAqk.data_ptr(),""",
    """    const int rc = g2_wy_s4_split_r(q_kgb.data_ptr(), k.data_ptr(), has_vb ? v_vb.data_ptr() : nullptr,
                                  has_vn ? v_new.data_ptr() : nullptr,
                                  g.data_ptr(), beta.data_ptr(), A.data_ptr(), h.data_ptr(),
                                  has_ando ? do_.data_ptr() : nullptr, dh.data_ptr(),
                                  has_dv ? dv.data_ptr() : nullptr,
                                  dAqk.data_ptr(),""")
sub(BI, """                                  has_ando ? nullptr : do_row->data_ptr(), cur_stream());""",
    """                                  has_ando ? nullptr : do_row->data_ptr(),
                                  has_dv ? nullptr : dv2_c->data_ptr(),
                                  has_vn ? nullptr : vn_c->data_ptr(), cur_stream());""")

# ---------------------------------------------------------------- python
sub(OPS, """def fwd_h_bn(wn, kn, un, gn_t, seg=None):
    \"\"\"P1 v8 backward-recompute form: emits bN(h) and Vn, no o.\"\"\"
    return _ext().fwd_h_bn(wn, kn, un, gn_t, seg)  # -> (HbN, Vn, ANv)""",
    """def fwd_h_bn(wn, kn, un, gn_t, seg=None, emit_an=True):
    \"\"\"P1 v8 backward-recompute form: emits bN(h) and Vn, no o.  emit_an=False (kra P-22):
    no aN(v_new) copy -- s4b rebuilds it from Vn.\"\"\"
    return _ext().fwd_h_bn(wn, kn, un, gn_t, seg, bool(emit_an))  # -> (HbN, Vn, ANv)""")
sub(OPS, """def dhu_bn(kga, qn, wb, dot, dvn, gn_t, scale, emit_dv2n=True, seg=None):""",
    """def dhu_bn(kga, qn, wb, dot, dvn, gn_t, scale, emit_dv2n=True, seg=None, emit_an2=True):""")
sub(OPS, """    return _ext().dhu_bn(kga, qn, wb, dot, dvn, gn_t, float(scale), bool(emit_dv2n), seg)""",
    """    return _ext().dhu_bn(kga, qn, wb, dot, dvn, gn_t, float(scale), bool(emit_dv2n), seg,
                         bool(emit_an2))""")
sub(OPS, """                q_row, v_row, dAqk, heads, tokens, scale, emit_dA=False, rowmap=None, do_row=None):""",
    """                q_row, v_row, dAqk, heads, tokens, scale, emit_dA=False, rowmap=None, do_row=None,
                vn_c=None, dv2_c=None):""")
sub(OPS, """                              None if do_row is None else do_row.reshape(-1).contiguous())""",
    """                              None if do_row is None else do_row.reshape(-1).contiguous(),
                              None if vn_c is None else vn_c.reshape(-1).contiguous(),
                              None if dv2_c is None else dv2_c.reshape(-1).contiguous())""")
sub(K3, """        if kept:                                   # R5 keep-state: bit-identical to the recompute
            bN_h, Vn, aN_vn = kept_bN, kept_Vn, kept_aN
            del kept_bN, kept_Vn, kept_aN
        else:
            bN_h, Vn, aN_vn = g2_ops.fwd_h_bn(Wn, Kn, Un, Gn, seg=ctx.seg)  # B2: P1 v8 (bN + aN direct)
        del Wn, Kn, Un""",
    """        # kra P-22: on the S4 split route the recurrences skip their aN copies; s4b / s4a rebuild
        # them from the C-form Vn / dV2n (KDA_G2_NOAN=0 restores them).  Keep-state keeps the
        # forward's saved aN_vn.
        no_an = no_vb and os.environ.get("KDA_G2_NOAN", "1").strip() != "0"
        if kept:                                   # R5 keep-state: bit-identical to the recompute
            bN_h, Vn, aN_vn = kept_bN, kept_Vn, kept_aN
            del kept_bN, kept_Vn, kept_aN
        else:
            bN_h, Vn, aN_vn = g2_ops.fwd_h_bn(Wn, Kn, Un, Gn, seg=ctx.seg,
                                              emit_an=not no_an)  # B2: P1 v8 (bN + aN direct)
        del Wn, Kn, Un
        vn_c = None
        if not kept and no_an:
            vn_c, aN_vn = Vn, torch.empty(0, dtype=Vn.dtype, device=Vn.device)""")
sub(K3, """        bN_dh, dV2n, aN_dv2 = g2_ops.dhu_bn(KgA, Qn, Wb, dOT, dVn, Gn, scale,
                                            emit_dv2n=emit_dv2n, seg=ctx.seg)
        del KgA, Qn, Wb, dOT, dVn, dV2n""",
    """        bN_dh, dV2n, aN_dv2 = g2_ops.dhu_bn(KgA, Qn, Wb, dOT, dVn, Gn, scale,
                                            emit_dv2n=emit_dv2n or no_an, seg=ctx.seg,
                                            emit_an2=not no_an)
        dv2_c = None
        if no_an:
            dv2_c, aN_dv2 = dV2n, torch.empty(0, dtype=dV2n.dtype, device=dV2n.device)
        del KgA, Qn, Wb, dOT, dVn, dV2n""")
sub(K3, """                                                             do_row=do if no_ando else None)""",
    """                                                             do_row=do if no_ando else None,
                                                             vn_c=vn_c, dv2_c=dv2_c)""")
print("P-22 APPLIED")
