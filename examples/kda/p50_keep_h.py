"""kra P-50: keep-state at the FLA-default memory budget (KDA_G2_KEEP_STATE=h).

The forward hands the backward ONLY bN(h) (512 B/token*head: the saved set becomes 520 + 512 =
1032 B/token*head = FLA default) instead of bN(h) + Vn + aN(v_new) (1544 B).  The backward skips
the fwd_h_bn recurrence and rebuilds v_new = U - W h for all chunks in parallel:
  * forward recurrence MODE 5 (Ob + HbN, no Vn / aN): compile-time instance, also for the wide kernel
  * g2_vnew_h_kernel: per (V-slab, head, chunk) CTA; hB is re-assembled from bN (the exact inverse
    of bN_emit), then fwd_h_v8o4's GEMM1 + rounding verbatim -> Vn bit-identical to fwd_h_bn's
  * wu emits Wn/Un for the v_new rebuild; it keeps emitting Kn (dropping it measured 708 vs 646 us on
    wu at 16K/12 -- a backward-only temporary, no effect on the saved set); emit_kn stays available
Every output is bit-identical to the recompute path.
    python3 p50_keep_h.py <tree>"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
F = ROOT / "csrc" / "g2" / "g2_fwd_v8.cuh"
B = ROOT / "csrc" / "g2" / "g2_bindings.cpp"
O = ROOT / "hip_kda" / "ops" / "g2_ops.py"
K = ROOT / "hip_kda" / "ops" / "k3_g2.py"


def rep(t, old, new, cnt=1):
    assert t.count(old) == cnt, (old[:100], t.count(old))
    return t.replace(old, new)


# ------------------------------------------------------------------ g2_fwd_v8.cuh
t = F.read_text()
t = rep(t, """    const bool hasVn = MODE == 0 ? p.Vn != nullptr : MODE != 1;
    const bool hasAN = MODE == 0 ? p.ANv != nullptr : (MODE != 1 && MODE != 4);""",
        """    const bool hasVn = MODE == 0 ? p.Vn != nullptr : (MODE != 1 && MODE != 5);   // kra P-50: 5 = Ob+HbN
    const bool hasAN = MODE == 0 ? p.ANv != nullptr : (MODE != 1 && MODE != 4 && MODE != 5);""", cnt=2)
t = rep(t, """    else if (O == nullptr && Ob == nullptr && Vn && HbN && !ANv) mode = 4;   // kra P-22""",
        """    else if (O == nullptr && Ob == nullptr && Vn && HbN && !ANv) mode = 4;   // kra P-22
    else if (O == nullptr && Ob != nullptr && !Vn && HbN && !ANv) mode = 5;  // kra P-50: keep-h forward""")
t = rep(t, """            case 4: launch4(g2_fwd_h_v8o4_kernel<VL, 4>); break;""",
        """            case 4: launch4(g2_fwd_h_v8o4_kernel<VL, 4>); break;
            case 5: launch4(g2_fwd_h_v8o4_kernel<VL, 5>); break;""")
t = rep(t, """                case 4: hipLaunchKernelGGL((g2_fwd_h_v8o4w_kernel<VL, 4>), gridw, dim3(256), 0, s, p); break;""",
        """                case 4: hipLaunchKernelGGL((g2_fwd_h_v8o4w_kernel<VL, 4>), gridw, dim3(256), 0, s, p); break;
                case 5: hipLaunchKernelGGL((g2_fwd_h_v8o4w_kernel<VL, 5>), gridw, dim3(256), 0, s, p); break;""")
vk = r"""
// kra P-50: keep-h backward -- v_new = U - W h for every chunk IN PARALLEL from the kept bN(h).
// One CTA per (head, chunk) covering the 4 V-slabs (W fetched once), 4 waves (wave = T-tile r1).
// Per slab, hB is re-assembled from bN with the exact inverse of bN_emit (bN(i) is the bf16 hB that
// fwd_h_v8o4's GEMM1 reads at step i), then GEMM1 and the v_new rounding run verbatim -> Vn
// bit-identical to the fwd_h_bn recompute.  The next slab's bN quads are in flight in registers
// during the current slab's GEMM1 (all-up-front measured slower: 400 vs 339 us at 16K/12); hB is
// double-buffered (one barrier per slab).
__global__ void __launch_bounds__(256) g2_vnew_h_kernel(const bf16* __restrict__ Wn,
                                                        const bf16* __restrict__ Un,
                                                        const bf16* __restrict__ HbN,
                                                        bf16* __restrict__ Vn, int H, int NT) {
    __shared__ __align__(16) bf16 hB[2][kBV * kPadK];
    const int tid = threadIdx.x, wv = tid >> 6, l = tid & 63;
    const int bh = blockIdx.x, i = blockIdx.y;
    const size_t chunk = (size_t)bh * NT;
    const int r1 = wv, m = l & 15, q4 = l >> 4;
    v4bh wf[8];
    {
        const bf16* src = Wn + (((chunk + i) * 4 + r1) * 8) * 256 + l * 8;
        #pragma unroll
        for (int j = 0; j < 4; ++j) ld16x2(src + j * 512, wf[2 * j], wf[2 * j + 1]);
    }
    auto loadH = [&](int vs, v4bh* hv) {               // fragment f = (kb, vt): 16 per slab
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int f = wv + 4 * j, kb = f >> 1, vt = f & 1;
            hv[j] = *reinterpret_cast<const v4bh*>(
                HbN + ((((size_t)i * H + bh) * 8 + kb) * 8 + (2 * vs + vt)) * 256 + l * 4);
        }
    };
    auto putH = [&](bf16* dst, const v4bh* hv) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int f = wv + 4 * j, kb = f >> 1, vt = f & 1;
            #pragma unroll
            for (int e = 0; e < 4; ++e)
                reinterpret_cast<short*>(dst)[(16 * vt + 4 * q4 + e) * kPadK + 16 * kb + pi16(m)] = hv[j][e];
        }
    };
    v4bh hA[4], hN[4];
    loadH(0, hA);
    #pragma unroll
    for (int vs = 0; vs < kK / kBV; ++vs) {
        bf16* hb = hB[vs & 1];
        putH(hb, hA);
        v4bh uf0, uf1;
        ld16x2(Un + (((chunk + i) * 4 + r1) * 8 + 2 * vs) * 256 + l * 8, uf0, uf1);
        if (vs + 1 < kK / kBV) loadH(vs + 1, hN);
        lds_barrier();
        f4 acc0 = f4{0.f,0.f,0.f,0.f}, acc1 = f4{0.f,0.f,0.f,0.f};
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
            acc0 = mmac(wf[kt], ld8(hb + (0 * 16 + m) * kPadK + 16 * kt + q4 * 4), acc0);
            acc1 = mmac(wf[kt], ld8(hb + (1 * 16 + m) * kPadK + 16 * kt + q4 * 4), acc1);
        }
        v4bh x0, x1;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float u0 = __uint_as_float(((uint32_t)(unsigned short)uf0[e]) << 16);
            const float u1 = __uint_as_float(((uint32_t)(unsigned short)uf1[e]) << 16);
            x0[e] = bf_rne(u0 - acc0[e]);
            x1[e] = bf_rne(u1 - acc1[e]);
        }
        *reinterpret_cast<v4bh*>(Vn + (((chunk + i) * 4 + r1) * 8 + 2 * vs + 0) * 256 + l * 4) = x0;
        *reinterpret_cast<v4bh*>(Vn + (((chunk + i) * 4 + r1) * 8 + 2 * vs + 1) * 256 + l * 4) = x1;
        #pragma unroll
        for (int j = 0; j < 4; ++j) hA[j] = hN[j];
    }
}

extern "C" int g2_vnew_from_h(const void* Wn, const void* Un, const void* HbN, void* Vn,
                              int H, int NT, hipStream_t s) {
    if (H <= 0 || NT <= 0) return -1;
    hipLaunchKernelGGL(g2_vnew_h_kernel, dim3((unsigned)H, (unsigned)NT), dim3(256), 0, s,
                       (const bf16*)Wn, (const bf16*)Un, (const bf16*)HbN, (bf16*)Vn, H, NT);
    return (int)hipGetLastError();
}
"""
anchor = 'extern "C" int g2_fwd_h_v8o(const void* Wn, const void* Kn, const void* Un, const void* Gn,'
t = rep(t, anchor, vk + "\n" + anchor)
F.write_text(t)

# ------------------------------------------------------------------ g2_bindings.cpp
t = B.read_text()
t = rep(t, """extern "C" int g2_dhu_v8b(const void* KgA, const void* Qg, const void* Wb, const void* dOT, const void* dv,""",
        """extern "C" int g2_vnew_from_h(const void* Wn, const void* Un, const void* HbN, void* Vn,
                              int H, int NT, hipStream_t s);   // kra P-50
extern "C" int g2_dhu_v8b(const void* KgA, const void* Qg, const void* Wb, const void* dOT, const void* dv,""")
t = rep(t, """std::vector<torch::Tensor> dhu_bn(torch::Tensor KgA, torch::Tensor Qn, torch::Tensor Wb,""",
        """// kra P-50: keep-h backward -- Vn = U - W h from the kept bN(h), all chunks in parallel
torch::Tensor vnew_from_h(torch::Tensor Wn, torch::Tensor Un, torch::Tensor HbN) {
    for (auto* t : {&Wn, &Un, &HbN}) { check_contig(*t, "vnew_from_h"); check_bf16(*t, "vnew_from_h"); }
    const int64_t H = Wn.size(1), NT = Wn.size(2);
    TORCH_CHECK(Un.numel() == Wn.numel() && HbN.numel() == H * NT * 16384, "vnew_from_h sizes");
    c10::hip::HIPGuard guard(Wn.get_device());
    auto Vn = torch::empty({1, H, NT, 4, 8, 4, 16, 4}, Wn.options());
    const int rc = g2_vnew_from_h(Wn.data_ptr(), Un.data_ptr(), HbN.data_ptr(), Vn.data_ptr(),
                                  (int)H, (int)NT, cur_stream());
    TORCH_CHECK(rc == 0, "g2_vnew_from_h failed: ", rc);
    return Vn;
}

std::vector<torch::Tensor> dhu_bn(torch::Tensor KgA, torch::Tensor Qn, torch::Tensor Wb,""")
t = rep(t, """                              c10::optional<torch::Tensor> clen, bool emit_p1, bool emit_vb) {""",
        """                              c10::optional<torch::Tensor> clen, bool emit_p1, bool emit_vb,
                              bool emit_kn) {""")
t = rep(t, """    auto Kn = emit_p1 ? torch::empty({1, heads, chunks, 8, 4, 4, 16, 4}, opt) : torch::Tensor();""",
        """    // kra P-50: emit_kn=false (keep-h backward): Wn/Un feed the v_new rebuild, Kn has no reader
    emit_kn = emit_kn && emit_p1;
    auto Kn = emit_kn ? torch::empty({1, heads, chunks, 8, 4, 4, 16, 4}, opt) : torch::Tensor();""")
t = rep(t, """                         emit_p1 ? Kn.data_ptr() : nullptr, Qn.data_ptr(),""",
        """                         emit_kn ? Kn.data_ptr() : nullptr, Qn.data_ptr(),""")
t = rep(t, """          py::arg("clen") = py::none(), py::arg("emit_p1") = true, py::arg("emit_vb") = true);""",
        """          py::arg("clen") = py::none(), py::arg("emit_p1") = true, py::arg("emit_vb") = true,
          py::arg("emit_kn") = true);
    m.def("vnew_from_h", &vnew_from_h, "G2 keep-h: Vn = U - W h from the kept bN(h) [kra P-50]");""")
B.write_text(t)

# ------------------------------------------------------------------ g2_ops.py
t = O.read_text()
t = rep(t, """def wu(k, v, q, beta, A, g, heads, chunks, rowmap=None, emit_p1=True, emit_vb=True):""",
        """def wu(k, v, q, beta, A, g, heads, chunks, rowmap=None, emit_p1=True, emit_vb=True, emit_kn=True):""")
t = rep(t, """                     *_rowmap(rowmap)[:2], bool(emit_p1), bool(emit_vb))""",
        """                     *_rowmap(rowmap)[:2], bool(emit_p1), bool(emit_vb), bool(emit_kn))


def vnew_from_h(wn, un, hbn):
    \"\"\"kra P-50 keep-h: Vn = U - W h for all chunks in parallel from the kept bN(h); bit-identical
    to the Vn the fwd_h_bn recompute writes.\"\"\"
    return _ext().vnew_from_h(wn, un, hbn)""")
O.write_text(t)

# ------------------------------------------------------------------ k3_g2.py
t = K.read_text()
t = rep(t, """def keep_state(ctx) -> bool:""", '''def keep_h(ctx) -> bool:
    """kra P-50: KDA_G2_KEEP_STATE=h -- the forward keeps ONLY bN(h) (512 B/token*head: saved set
    1032 B = FLA default) whenever a backward follows; the backward rebuilds v_new in parallel."""
    return os.environ.get("KDA_G2_KEEP_STATE", "auto").strip() == "h" and any(ctx.needs_input_grad)


def keep_state(ctx) -> bool:''')
t = rep(t, """    mode = os.environ.get("KDA_G2_KEEP_STATE", "auto").strip()
    if mode == "0" or not any(ctx.needs_input_grad):""",
        """    mode = os.environ.get("KDA_G2_KEEP_STATE", "auto").strip()
    if mode in ("0", "h") or not any(ctx.needs_input_grad):""")
old_fwd = """            keep = keep_state(ctx)
            outs = g2_ops.fwd_h_o(Wn, Kn, Un, Gn, QgA, AqkA, scale, emit_bn=keep, o_bf16=obf,
                                  seg=seg, rowmap=lay, emit_vn=keep, emit_anv=keep)
            o = outs[0]
            kept = (outs[2], outs[1], outs[3]) if keep else (None, None, None)   # bN_h, Vn, aN_vn"""
new_fwd = """            keep = keep_state(ctx)
            kh = keep_h(ctx)                                 # kra P-50: bN(h) only
            outs = g2_ops.fwd_h_o(Wn, Kn, Un, Gn, QgA, AqkA, scale, emit_bn=keep or kh, o_bf16=obf,
                                  seg=seg, rowmap=lay, emit_vn=keep, emit_anv=keep)
            o = outs[0]
            kept = ((outs[2], outs[1], outs[3]) if keep else
                    (outs[2], None, None) if kh else (None, None, None))   # bN_h, Vn, aN_vn"""
t = rep(t, old_fwd, new_fwd)
old_fwd2 = """        keep = keep_state(ctx)
        outs = g2_ops.fwd_h_o(Wn, Kn, Un, Gn, QgA, AqkA, scale, emit_bn=keep, o_bf16=obf,
                              seg=seg, rowmap=lay, emit_vn=keep, emit_anv=keep)
        o = outs[0]
        kept = (outs[2], outs[1], outs[3]) if keep else (None, None, None)   # bN_h, Vn, aN_vn"""
new_fwd2 = """        keep = keep_state(ctx)
        kh = keep_h(ctx)                                     # kra P-50: bN(h) only
        outs = g2_ops.fwd_h_o(Wn, Kn, Un, Gn, QgA, AqkA, scale, emit_bn=keep or kh, o_bf16=obf,
                              seg=seg, rowmap=lay, emit_vn=keep, emit_anv=keep)
        o = outs[0]
        kept = ((outs[2], outs[1], outs[3]) if keep else
                (outs[2], None, None) if kh else (None, None, None))   # bN_h, Vn, aN_vn"""
t = rep(t, old_fwd2, new_fwd2)
t = rep(t, """        kept = kept_bN is not None                 # R5: the forward handed over the B2 state""",
        """        kept = kept_bN is not None                 # R5: the forward handed over the B2 state
        kept_h = kept and kept_Vn is None          # kra P-50: bN(h) only -> rebuild v_new""")
t = rep(t, """        Wn, Wb, Un, KgA, Kn, Qn, kgB, vB = g2_ops.wu(kn, v, qn, beta, Akk, g, H, NT, rowmap=lay,
                                                     emit_p1=not kept, emit_vb=not no_vb)""",
        """        no_an = no_vb and os.environ.get("KDA_G2_NOAN", "1").strip() != "0"
        h_only = kept_h and no_an                  # kra P-50 (the aN route falls back to fwd_h_bn)
        Wn, Wb, Un, KgA, Kn, Qn, kgB, vB = g2_ops.wu(kn, v, qn, beta, Akk, g, H, NT, rowmap=lay,
                                                     emit_p1=(not kept) or kept_h, emit_vb=not no_vb)""")
t = rep(t, """        no_an = no_vb and os.environ.get("KDA_G2_NOAN", "1").strip() != "0"
        if kept:                                   # R5 keep-state: bit-identical to the recompute
            bN_h, Vn, aN_vn = kept_bN, kept_Vn, kept_aN
            del kept_bN, kept_Vn, kept_aN
        else:
            bN_h, Vn, aN_vn = g2_ops.fwd_h_bn(Wn, Kn, Un, Gn, seg=ctx.seg,
                                              emit_an=not no_an)  # B2: P1 v8 (bN + aN direct)
        del Wn, Kn, Un
        vn_c = None
        if not kept and no_an:""", """        if kept and not kept_h:                    # R5 keep-state: bit-identical to the recompute
            bN_h, Vn, aN_vn = kept_bN, kept_Vn, kept_aN
            del kept_bN, kept_Vn, kept_aN
        elif h_only:                               # kra P-50: Vn = U - W h, chunk-parallel
            bN_h, aN_vn = kept_bN, None
            Vn = g2_ops.vnew_from_h(Wn, Un, bN_h)
            del kept_bN
        else:
            bN_h, Vn, aN_vn = g2_ops.fwd_h_bn(Wn, Kn, Un, Gn, seg=ctx.seg,
                                              emit_an=not no_an)  # B2: P1 v8 (bN + aN direct)
        del Wn, Kn, Un
        vn_c = None
        if (not kept or h_only) and no_an:""")
K.write_text(t)
print("P-50 APPLIED")
