"""kra P-17 (global route, round 3, structural): drop the vB layout copy.  s4a already loads the
row-major v (v_row, 8 x 8 B per lane, for db); with v_vb == nullptr it stages those values in S
(dvT is dead right after the dv2/db loop; a [64][136] bf16 tile fills S exactly) and the dA
MMAC reads its B fragments (row 16*st + pi16(m), columns 16*vt + 4*q) from S instead of the
vB-layout global copy.  wu skips the scattered vB emit when vB == nullptr.  The production
route (S4=1, P4=3) passes nullptr; every other route keeps vB.  Values and MMAC order are
unchanged -> bit-identical.        Usage: python3 p17_s4a_v_onchip.py <tree root>"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])


def sub(path, old, new, count=1):
    t = path.read_text()
    assert t.count(old) == count, (path.name, old[:90], t.count(old))
    path.write_text(t.replace(old, new))


WY = ROOT / "csrc/g2/g2_wy.cuh"
WU = ROOT / "csrc/g2/g2_wu.cuh"
BI = ROOT / "csrc/g2/g2_bindings.cpp"
OPS = ROOT / "hip_kda/ops/g2_ops.py"
K3 = ROOT / "hip_kda/ops/k3_g2.py"

# ---------------------------------------------------------------- s4a: v fragments from S
sub(WY, """        float db = 0.f;
        #pragma unroll
        for (int vt = 0; vt < 8; ++vt) {
            f4 acc = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int st = 0; st < 4; ++st) acc = mmac(aa[st], ld8(S + (16 * vt + pm) * kPT + 16 * st + 4 * q), acc);
            const v4bh vv = trok ? ld8(v_ + ro + 16 * vt) : v4bh{0, 0, 0, 0};""",
    """        float db = 0.f;
        v4bh vrow[8];                // kra P-17: this lane's row-major v quads, kept for the dA MMAC
        #pragma unroll
        for (int vt = 0; vt < 8; ++vt) {
            f4 acc = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int st = 0; st < 4; ++st) acc = mmac(aa[st], ld8(S + (16 * vt + pm) * kPT + 16 * st + 4 * q), acc);
            const v4bh vv = trok ? ld8(v_ + ro + 16 * vt) : v4bh{0, 0, 0, 0};
            vrow[vt] = vv;""")
sub(WY, """        f4 dA[4];
        #pragma unroll
        for (int st = 0; st < 4; ++st) dA[st] = f4{0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int vt = 0; vt < 8; ++vt)
            #pragma unroll
            for (int st = 0; st < 4; ++st)
                dA[st] = mmac(dva[vt], ld8(vBN + ((st * 8 + vt) * 64 + l) * 4), dA[st]);
        lds_barrier();                                                      // dvT dead -> dwT""",
    """        f4 dA[4];
        #pragma unroll
        for (int st = 0; st < 4; ++st) dA[st] = f4{0.f, 0.f, 0.f, 0.f};
        if (p.v_vb != nullptr) {
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt)
                #pragma unroll
                for (int st = 0; st < 4; ++st)
                    dA[st] = mmac(dva[vt], ld8(vBN + ((st * 8 + vt) * 64 + l) * 4), dA[st]);
        } else {
            // kra P-17: no vB copy -- the B fragment of (st, vt) is v[16*st + pi16(m)][16*vt+4q..+3],
            // i.e. the row-major quads the CTA just loaded; stage them in S ([64][136] = all of S,
            // free now that every wave finished reading dvT) and read the fragments from LDS.
            constexpr int kVS = 2 * kPT;                            // 136-short rows
            lds_barrier();                                          // dvT reads done
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt)
                *reinterpret_cast<v4bh*>(reinterpret_cast<short*>(S) + tr * kVS + 16 * vt + 4 * q) = vrow[vt];
            lds_barrier();                                          // v tile visible
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt)
                #pragma unroll
                for (int st = 0; st < 4; ++st)
                    dA[st] = mmac(dva[vt], ld8(S + (16 * st + pm) * kVS + 16 * vt + 4 * q), dA[st]);
        }
        lds_barrier();                                                      // dvT/v tile dead -> dwT""")

# ---------------------------------------------------------------- wu: vB emit optional
sub(WU, """            *reinterpret_cast<v4bh*>(p.vB + wu_vb_quad(j, vq, nblk3)) = vv;
        }
        if (p.stop == 4) return;""",
    """            if (p.vB != nullptr)   // kra P-17: production route reads v row-major in s4a instead
                *reinterpret_cast<v4bh*>(p.vB + wu_vb_quad(j, vq, nblk3)) = vv;
        }
        if (p.stop == 4) return;""")

# ---------------------------------------------------------------- bindings
sub(BI, """                              c10::optional<torch::Tensor> clen, bool emit_p1) {
    TORCH_CHECK(k.dim() == 4 && k.size(0) == 1, "g2_wu: B=1 only (batch = varlen rows)");""",
    """                              c10::optional<torch::Tensor> clen, bool emit_p1, bool emit_vb) {
    TORCH_CHECK(k.dim() == 4 && k.size(0) == 1, "g2_wu: B=1 only (batch = varlen rows)");""")
sub(BI, """    auto vB = torch::empty({1, chunks, heads, 4, 8, 4, 16, 4}, opt);
    const int rc = g2_wu(""",
    """    // kra P-17: emit_vb=false (production S4 split route): s4a reads v row-major, no vB copy
    auto vB = emit_vb ? torch::empty({1, chunks, heads, 4, 8, 4, 16, 4}, opt) : torch::Tensor();
    const int rc = g2_wu(""")
sub(BI, """                         kgB.data_ptr(), vB.data_ptr(),
                         1, (int)(chunks * 64), (int)heads, rm.ctok, rm.clen, cur_stream());""",
    """                         kgB.data_ptr(), emit_vb ? vB.data_ptr() : nullptr,
                         1, (int)(chunks * 64), (int)heads, rm.ctok, rm.clen, cur_stream());""")
sub(BI, """          py::arg("clen") = py::none(), py::arg("emit_p1") = true);
    m.def("do_dual", &do_dual, "G2 dO dual-layout single launch (S3-11)");""",
    """          py::arg("clen") = py::none(), py::arg("emit_p1") = true, py::arg("emit_vb") = true);
    m.def("do_dual", &do_dual, "G2 dO dual-layout single launch (S3-11)");""")
sub(BI, """    for (auto* t : {&q_kgb, &k, &v_vb, &v_new, &A, &h, &do_, &dh, &dv, &q_row, &v_row}) {
        check_contig(*t, "wy_s4_split input"); check_bf16(*t, "wy_s4_split input");
    }""",
    """    // kra P-17: an EMPTY v_vb selects the no-copy route (s4a stages the row-major v on chip)
    const bool has_vb = v_vb.numel() > 0;
    for (auto* t : {&q_kgb, &k, &v_new, &A, &h, &do_, &dh, &dv, &q_row, &v_row}) {
        check_contig(*t, "wy_s4_split input"); check_bf16(*t, "wy_s4_split input");
    }
    if (has_vb) { check_contig(v_vb, "wy_s4_split v_vb"); check_bf16(v_vb, "wy_s4_split v_vb"); }""")
sub(BI, """                want_nat(do_) && want_nat(dv) && want_nat(q_kgb) && want_nat(v_vb) &&""",
    """                want_nat(do_) && want_nat(dv) && want_nat(q_kgb) && (!has_vb || want_nat(v_vb)) &&""")
sub(BI, """    const int rc = g2_wy_s4_split(q_kgb.data_ptr(), k.data_ptr(), v_vb.data_ptr(), v_new.data_ptr(),""",
    """    const int rc = g2_wy_s4_split(q_kgb.data_ptr(), k.data_ptr(), has_vb ? v_vb.data_ptr() : nullptr,
                                  v_new.data_ptr(),""")

# ---------------------------------------------------------------- python
sub(OPS, """def wu(k, v, q, beta, A, g, heads, chunks, rowmap=None, emit_p1=True):""",
    """def wu(k, v, q, beta, A, g, heads, chunks, rowmap=None, emit_p1=True, emit_vb=True):""")
sub(OPS, """                     *_rowmap(rowmap)[:2], bool(emit_p1))""",
    """                     *_rowmap(rowmap)[:2], bool(emit_p1), bool(emit_vb))""")
sub(K3, """        Wn, Wb, Un, KgA, Kn, Qn, kgB, vB = g2_ops.wu(kn, v, qn, beta, Akk, g, H, NT, rowmap=lay,
                                                     emit_p1=not kept)""",
    """        # kra P-17: on the default S4 split route (S4=1, P4=3) s4a reads v row-major and builds its
        # dA operand on chip, so wu skips the vB copy (KDA_G2_NOVB=0 restores it).
        no_vb = (os.environ.get("KDA_G2_S4", "1").strip() != "0"
                 and os.environ.get("KDA_G2_P4", "3").strip() == "3"
                 and os.environ.get("KDA_G2_NOVB", "1").strip() != "0")
        Wn, Wb, Un, KgA, Kn, Qn, kgB, vB = g2_ops.wu(kn, v, qn, beta, Akk, g, H, NT, rowmap=lay,
                                                     emit_p1=not kept, emit_vb=not no_vb)
        if no_vb:
            vB = torch.empty(0, dtype=v.dtype, device=v.device)   # empty v_vb -> on-chip route""")
print("P-17 APPLIED")
