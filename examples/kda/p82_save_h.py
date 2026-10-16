"""kra P-82 (bitwise; user-approved memory: +512 B/token*head saved, total = FLA default): option d of
results/kda-segfusion-20261016 §8 -- the forward saves h, the backward drops its recompute recurrence.

  A. forward recurrence MODE 5 = Ob + bN(h) (P-81a), instances for every v8o4 path (sg+kp: MODE 1's production
     template + BP; kp: as MODE 2's kp instance; else <VL, 5>) and the wide v8o4w form.
  B. wu VNEW: given the saved bN(h) (HbN) it keeps W's A fragments (we8 of the w phase == fwd_h_bn's wf, same wave /
     lane) and, in the u phase, forms v_new = bf_rne(u - W h) per V-slab exactly as fwd_h_bn's GEMM1 does: hB (v rows,
     natural k) is rebuilt in the free slab from bN by bN_emit's inverse map (hB[16vt + 4q + e][16kb + pi16(m)] =
     bN[kb][2vs + vt][l][e]), acc0/acc1 = the kt = 0..7 MMAC chains, u from the u phase's bf16 C fragments; Vn is
     written in fwd_h_bn's layout.  Two V-slabs per slab fill (2 x 32 x 132 bf16 = 16.9 KB <= 17.4 KB).
  C. k3_g2: the forward (R4 route) emits bN when a backward follows (KDA_G2_SAVE_H, default on; keep-state keeps its
     MODE 2), the backward passes it to wu (emit_p1=False) and skips fwd_h_bn.
    python3 p82_save_h.py <tree>"""
import sys
from pathlib import Path

R = Path(sys.argv[1])
G = R / "csrc" / "g2"


def sub(path, old, new, cnt=1):
    s = path.read_text()
    assert s.count(old) == cnt, (path.name, old[:90], s.count(old))
    path.write_text(s.replace(old, new))


# ---------------------------------------------------------------- A. forward MODE 5
fv = G / "g2_fwd_v8.cuh"
sub(fv, """    const bool hasVn = MODE == 0 ? p.Vn != nullptr : MODE != 1;
    const bool hasAN = MODE == 0 ? p.ANv != nullptr : (MODE != 1 && MODE != 4);""",
    """    const bool hasVn = MODE == 0 ? p.Vn != nullptr : (MODE != 1 && MODE != 5);   // kra P-82: MODE 5 = Ob + HbN
    const bool hasAN = MODE == 0 ? p.ANv != nullptr : (MODE != 1 && MODE != 4 && MODE != 5);""", cnt=2)
sub(fv, """    else if (O == nullptr && Ob == nullptr && Vn && HbN && !ANv) mode = 4;   // kra P-22""",
    """    else if (O == nullptr && Ob == nullptr && Vn && HbN && !ANv) mode = 4;   // kra P-22
    else if (O == nullptr && Ob != nullptr && !Vn && HbN && !ANv) mode = 5;   // kra P-82: forward saves h""")
sub(fv, """            case 1: if (sg) { if (kp) launch4(g2_fwd_h_v8o4_kernel<VL, 1, true, true, true, true, true>);   // kra P-60""",
    """            case 5: if (sg && kp) launch4(g2_fwd_h_v8o4_kernel<VL, 5, true, true, true, true, true, true>);   // kra P-82
                    else if (kp) launch4(g2_fwd_h_v8o4_kernel<VL, 5, true, false, false, true, true, true>);
                    else    launch4(g2_fwd_h_v8o4_kernel<VL, 5>);
                    break;
            case 1: if (sg) { if (kp) launch4(g2_fwd_h_v8o4_kernel<VL, 1, true, true, true, true, true>);   // kra P-60""")
sub(fv, """                case 4: hipLaunchKernelGGL((g2_fwd_h_v8o4w_kernel<VL, 4>), gridw, dim3(256), 0, s, p); break;""",
    """                case 4: hipLaunchKernelGGL((g2_fwd_h_v8o4w_kernel<VL, 4>), gridw, dim3(256), 0, s, p); break;
                case 5: hipLaunchKernelGGL((g2_fwd_h_v8o4w_kernel<VL, 5>), gridw, dim3(256), 0, s, p); break;   // kra P-82""")

# ---------------------------------------------------------------- B. wu VNEW
wu = G / "g2_wu.cuh"
sub(wu, """    int kp = 0;    // kra P-61e: Kn lane-permuted (H <= KPERM_MAX_H)
};""", """    int kp = 0;    // kra P-61e: Kn lane-permuted (H <= KPERM_MAX_H)
    const short* HbN = nullptr;   // kra P-82: saved bN(h) -> v_new = u - W h formed here (VnOut)
    short* VnOut = nullptr;
};""")
sub(wu, "g2_glb(p_.ctok), g2_glb(p_.clen), p_.stop, p_.kp};",
    "g2_glb(p_.ctok), g2_glb(p_.clen), p_.stop, p_.kp, g2_glb(p_.HbN), g2_glb(p_.VnOut)};   // kra P-82")
sub(wu, """        auto mmac_phase = [&](short* outA, short* outB, short* outC, int stop_acc) {""",
    """        v4bh wA[8];   // kra P-82: the w phase's A fragments (== fwd_h_bn's wf for this wave / lane)
        auto mmac_phase = [&](short* outA, short* outB, short* outC, int stop_acc) {""")
sub(wu, """            v4bh we8[8];
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt)
                #pragma unroll
                for (int e = 0; e < 4; ++e) we8[vt][e] = bf_rne(acc[vt][e]);
""", """            v4bh we8[8];
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt)
                #pragma unroll
                for (int e = 0; e < 4; ++e) we8[vt][e] = bf_rne(acc[vt][e]);
            if (p.VnOut != nullptr && stop_acc == 2) {   // kra P-82: keep W for v_new
                #pragma unroll
                for (int vt = 0; vt < 8; ++vt) wA[vt] = we8[vt];
            }
            if (p.VnOut != nullptr && stop_acc == 5) {   // kra P-82: v_new = u - W h (fwd_h_bn's GEMM1 + rounding)
                short* hb = slab;                        // 2 x [32 v][kPadK] hB tiles
                const int m = l & 15, q4 = l >> 4;
                #pragma unroll   // static we8 indices (R55: a runtime vs puts we8 in scratch)
                for (int pr = 0; pr < 2; ++pr) {
                    __syncthreads();                     // slab free: MMAC reads / previous pair's reads done
                    #pragma unroll
                    for (int it = 0; it < 8; ++it) {     // 2 slabs x 8 kb x 2 vt x 64 lanes v4bh / 256 threads
                        const int jj = tid + 256 * it;
                        const int ln = jj & 63, vt = (jj >> 6) & 1, kb = (jj >> 7) & 7, sp = jj >> 10;
                        const int vs = 2 * pr + sp;
                        const v4bh val = *reinterpret_cast<const v4bh*>(
                            p.HbN + ((((size_t)i * p.H + hh) * 8 + kb) * 8 + (2 * vs + vt)) * 256 + ln * 4);
                        const int qq = ln >> 4, mm = ln & 15;
                        #pragma unroll
                        for (int e = 0; e < 4; ++e)
                            hb[sp * 32 * kPadK + (16 * vt + 4 * qq + e) * kPadK + 16 * kb + pi16(mm)] = val[e];
                    }
                    __syncthreads();
                    #pragma unroll
                    for (int sp = 0; sp < 2; ++sp) {
                        const int vs = 2 * pr + sp;
                        const short* hs = hb + sp * 32 * kPadK;
                        f4 acc0 = f4{0.f, 0.f, 0.f, 0.f}, acc1 = f4{0.f, 0.f, 0.f, 0.f};
                        #pragma unroll
                        for (int kt = 0; kt < 8; ++kt) {
                            const v4bh hbf0 = *reinterpret_cast<const v4bh*>(hs + (0 * 16 + m) * kPadK + 16 * kt + q4 * 4);
                            const v4bh hbf1 = *reinterpret_cast<const v4bh*>(hs + (1 * 16 + m) * kPadK + 16 * kt + q4 * 4);
                            acc0 = mmac(wA[kt], hbf0, acc0);
                            acc1 = mmac(wA[kt], hbf1, acc1);
                        }
                        v4bh x0, x1;
                        #pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            const float u0 = __uint_as_float(((uint32_t)(unsigned short)we8[2 * vs][e]) << 16);
                            const float u1 = __uint_as_float(((uint32_t)(unsigned short)we8[2 * vs + 1][e]) << 16);
                            x0[e] = bf_rne(u0 - acc0[e]);
                            x1[e] = bf_rne(u1 - acc1[e]);
                        }
                        *reinterpret_cast<v4bh*>(p.VnOut + nblk + (size_t)(8 * wv + 2 * vs + 0) * 256 + l * 4) = x0;
                        *reinterpret_cast<v4bh*>(p.VnOut + nblk + (size_t)(8 * wv + 2 * vs + 1) * 256 + l * 4) = x1;
                    }
                }
                __syncthreads();                         // slab reuse by the next phase
            }
""")
sub(wu, """                     int B, int T, int H, const int* ctok, const int* clen, hipStream_t s) {
    // T = chunk-padded rows (NT = T/64); with ctok/clen the row inputs are packed.""",
    """                     int B, int T, int H, const int* ctok, const int* clen,
                     const void* hbn, void* vn_out, hipStream_t s) {   // kra P-82
    // T = chunk-padded rows (NT = T/64); with ctok/clen the row inputs are packed.""")
sub(wu, """    p.kp = H <= g2::g2_env_long("KDA_G2_KPERM_MAX_H", 20);   // kra P-61e
""", """    p.kp = H <= g2::g2_env_long("KDA_G2_KPERM_MAX_H", 20);   // kra P-61e
    p.HbN = (const short*)hbn; p.VnOut = (short*)vn_out;      // kra P-82
""")

bd = G / "g2_bindings.cpp"
sub(bd, """                     void* kgB, void* vB,
                     int B, int T, int H, const int* ctok, const int* clen, hipStream_t s);""",
    """                     void* kgB, void* vB,
                     int B, int T, int H, const int* ctok, const int* clen,
                     const void* hbn, void* vn_out, hipStream_t s);   // kra P-82""")
sub(bd, """                              c10::optional<torch::Tensor> clen, bool emit_p1, bool emit_vb) {
    TORCH_CHECK(k.dim() == 4 && k.size(0) == 1, "g2_wu: B=1 only (batch = varlen rows)");""",
    """                              c10::optional<torch::Tensor> clen, bool emit_p1, bool emit_vb,
                              c10::optional<torch::Tensor> hbn) {   // kra P-82
    TORCH_CHECK(k.dim() == 4 && k.size(0) == 1, "g2_wu: B=1 only (batch = varlen rows)");
    const bool vnew = hbn.has_value() && hbn->defined();
    if (vnew) { check_contig(*hbn, "wu hbn"); check_bf16(*hbn, "wu hbn"); }""")
sub(bd, """    auto vB = emit_vb ? torch::empty({1, chunks, heads, 4, 8, 4, 16, 4}, opt) : torch::Tensor();
    const int rc = g2_wu(""", """    auto vB = emit_vb ? torch::empty({1, chunks, heads, 4, 8, 4, 16, 4}, opt) : torch::Tensor();
    auto Vn = vnew ? torch::empty({1, heads, chunks, 4, 8, 4, 16, 4}, opt) : torch::Tensor();   // kra P-82
    const int rc = g2_wu(""")
sub(bd, """                         1, (int)(chunks * 64), (int)heads, rm.ctok, rm.clen, cur_stream());
    TORCH_CHECK(rc == 0, "g2_wu failed: ", rc);
    return {Wn, Wb, Un, KgA, Kn, Qn, kgB, vB};""",
    """                         1, (int)(chunks * 64), (int)heads, rm.ctok, rm.clen,
                         vnew ? hbn->data_ptr() : nullptr, vnew ? Vn.data_ptr() : nullptr, cur_stream());
    TORCH_CHECK(rc == 0, "g2_wu failed: ", rc);
    return {Wn, Wb, Un, KgA, Kn, Qn, kgB, vB, Vn};""")
sub(bd, """          py::arg("clen") = py::none(), py::arg("emit_p1") = true, py::arg("emit_vb") = true);""",
    """          py::arg("clen") = py::none(), py::arg("emit_p1") = true, py::arg("emit_vb") = true,
          py::arg("hbn") = py::none());   // kra P-82""")

po = R / "hip_kda" / "ops" / "g2_ops.py"
sub(po, """def wu(k, v, q, beta, A, g, heads, chunks, rowmap=None, emit_p1=True, emit_vb=True):""",
    """def wu(k, v, q, beta, A, g, heads, chunks, rowmap=None, emit_p1=True, emit_vb=True, hbn=None):""")
sub(po, """    return _ext().wu(k.contiguous(), v.contiguous(), q.contiguous(), beta.contiguous(),
                     A.contiguous(), g.contiguous(), int(heads), int(chunks),
                     *_rowmap(rowmap)[:2], bool(emit_p1), bool(emit_vb))""",
    """    r = _ext().wu(k.contiguous(), v.contiguous(), q.contiguous(), beta.contiguous(),
                  A.contiguous(), g.contiguous(), int(heads), int(chunks),
                  *_rowmap(rowmap)[:2], bool(emit_p1), bool(emit_vb),
                  None if hbn is None else hbn.contiguous())
    return list(r) if hbn is not None else list(r)[:8]   # kra P-82: + Vn (v_new from the saved h)""")

# ---------------------------------------------------------------- C. k3_g2
k3 = R / "hip_kda" / "ops" / "k3_g2.py"
sub(k3, """            keep = keep_state(ctx)
            outs = g2_ops.fwd_h_o(Wn, Kn, Un, Gn, QgA, AqkA, scale, emit_bn=keep, o_bf16=obf,
                                  seg=seg, rowmap=lay, emit_vn=keep, emit_anv=keep)
            o = outs[0]
            kept = (outs[2], outs[1], outs[3]) if keep else (None, None, None)   # bN_h, Vn, aN_vn""",
    """            keep = keep_state(ctx)
            # kra P-82: save h (bN) whenever a backward follows -> the backward drops its recompute
            # (H <= KDA_G2_SAVE_H_MAX_H, default 20 = the kp/BP instances; at H = 48 / 96 the forward bN emit
            # costs +51% of the recurrence and the save does not pay)
            hsave = (not keep and any(ctx.needs_input_grad)
                     and os.environ.get("KDA_G2_SAVE_H", "1").strip() != "0"
                     and H <= int(os.environ.get("KDA_G2_SAVE_H_MAX_H", "20")))
            outs = g2_ops.fwd_h_o(Wn, Kn, Un, Gn, QgA, AqkA, scale, emit_bn=keep or hsave, o_bf16=obf,
                                  seg=seg, rowmap=lay, emit_vn=keep, emit_anv=keep)
            o = outs[0]
            kept = ((outs[2], outs[1], outs[3]) if keep
                    else ((outs[2], None, None) if hsave else (None, None, None)))   # bN_h, Vn, aN_vn""")
sub(k3, """        kept = kept_bN is not None                 # R5: the forward handed over the B2 state""",
    """        # kra P-82: h-only save (forward MODE 5) -> wu forms v_new, no recompute recurrence
        no_an_e = (os.environ.get("KDA_G2_S4", "1").strip() != "0"
                   and os.environ.get("KDA_G2_P4", "3").strip() == "3"
                   and os.environ.get("KDA_G2_NOVB", "1").strip() != "0"
                   and os.environ.get("KDA_G2_NOAN", "1").strip() != "0")
        hsave = kept_bN is not None and kept_Vn is None
        if hsave and not no_an_e:
            raise RuntimeError("k3_g2: the saved-h backward needs the default S4/NOAN route")
        kept = kept_bN is not None and not hsave   # R5: the forward handed over the B2 state""")
sub(k3, """        Wn, Wb, Un, KgA, Kn, Qn, kgB, vB = g2_ops.wu(kn, v, qn, beta, Akk, g, H, NT, rowmap=lay,
                                                     emit_p1=not kept, emit_vb=not no_vb)""",
    """        wo = g2_ops.wu(kn, v, qn, beta, Akk, g, H, NT, rowmap=lay,
                       emit_p1=not kept and not hsave, emit_vb=not no_vb,
                       hbn=kept_bN if hsave else None)   # kra P-82
        Wn, Wb, Un, KgA, Kn, Qn, kgB, vB = wo[:8]
        Vn_w = wo[8] if hsave else None
        del wo""")
sub(k3, """        if kept:                                   # R5 keep-state: bit-identical to the recompute
            bN_h, Vn, aN_vn = kept_bN, kept_Vn, kept_aN
            del kept_bN, kept_Vn, kept_aN
        else:""", """        if kept:                                   # R5 keep-state: bit-identical to the recompute
            bN_h, Vn, aN_vn = kept_bN, kept_Vn, kept_aN
            del kept_bN, kept_Vn, kept_aN
        elif hsave:                                # kra P-82: saved h + wu's v_new (bit-identical)
            bN_h, Vn, aN_vn = kept_bN, Vn_w, None
            del kept_bN, Vn_w
        else:""")
sub(k3, """        vn_c = None
        if not kept and no_an:
            vn_c, aN_vn = Vn, torch.empty(0, dtype=Vn.dtype, device=Vn.device)""",
    """        vn_c = None
        if not kept and no_an:   # kra P-82: also the saved-h route (no aN copy)
            vn_c, aN_vn = Vn, torch.empty(0, dtype=Vn.dtype, device=Vn.device)""")
print("P-82 APPLIED")
