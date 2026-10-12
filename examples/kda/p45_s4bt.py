"""kra P-45 = B4 step 1 (round 13, B-class): s4b + bwd_tail in ONE kernel (KDA_G2_S4BT=1).

The production 4-wave / 2-CTA s4b computes dq/dk/dg tiles per kt; the tail (reverse chunk-cumsum of dg
+ gate backward, l2norm backward of dq/dk, beta passthrough, dA_log / ddt_bias partials) used to be a
separate kernel re-reading fp32 dq/dk/dg.  Here, inside the s4b kt loop (compile-time TAIL instance):
  * dg never goes to HBM: the reverse suffix over the wave's 16 rows is a 4-step shfl scan, the wave's
    block total is parked in the table pad rows (TA rows 64..67, never touched by the tables), the
    carry from the blocks below is read after the existing end-of-kt barrier, then the gate backward
    (FLA operation order) writes bf16 dg_raw; ddt / dA partials per (wave, column / CTA)
  * the l2norm row dots accumulate in two fp32 registers
  * after the loop each lane re-reads its own fp32 dq/dk and qn/kn and writes bf16 dq/dk
  * db goes straight to the returned (fp32) db (use_beta = false: passthrough)
The fixed-order association of the scan / dots / partials differs from the tail kernel (deterministic,
B-class gate).  Scope: gate on (not precomputed), qk l2norm on, beta sigmoid off (the K3 call).
python3 p45_s4bt.py <tree>"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
P = root / "csrc" / "g2" / "g2_wy.cuh"
t = P.read_text()


def rep(old, new, count=1):
    global t
    assert t.count(old) == count, (old[:90], t.count(old))
    t = t.replace(old, new)


R0 = "template <int A, int M>\n__device__ __forceinline__ void s4b_tail("
R1 = "template <bool VL>\n__global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(1, 3)))\np3_wy_s4b_kernel(WyS4BParams p) {"


def rep_s4b(old, new):
    """replace exactly once inside s4b_tail's body (the old fused s4 kernel has the same loop text)"""
    global t
    a = t.index(R0) if R0 in t else t.index(R0.replace("<int A, int M>", "<int A, int M, bool TAIL = false>"))
    b = t.index(R1) if R1 in t else t.index(R1.replace("<bool VL>", "<bool VL, bool TAIL = false>"))
    reg = t[a:b]
    assert reg.count(old) == 1, (old[:90], reg.count(old))
    t = t[:a] + reg.replace(old, new) + t[b:]


# ---- params
rep("""    const bf16* do_row;                 // kra P-20: row-major do, read when do_ (aN_do) is null
    const bf16* vnc;                    // kra P-22: C-form v_new (std order), read when v_new is null
};""", """    const bf16* do_row;                 // kra P-20: row-major do, read when do_ (aN_do) is null
    const bf16* vnc;                    // kra P-22: C-form v_new (std order), read when v_new is null
    // kra P-45 (s4b + tail fusion, TAIL instance only)
    const float *q_rstd, *k_rstd;
    const short* g_raw; const float *A_log, *dt_bias;
    short *dqo, *dko, *dg_raw;
    float* db_out;
    float *dA_part4, *ddt_part4;        // [H][NT][4 waves], [H*128][NT][4 waves]
    float lower_bound;
};""")
# ---- s4b_tail: template flag + tail work
rep("""template <int A, int M>
__device__ __forceinline__ void s4b_tail(const WyS4BParams& p, S4BSmem& sm, const v4bh (&aa)[4],""",
    """template <int A, int M, bool TAIL = false>
__device__ __forceinline__ void s4b_tail(const WyS4BParams& p, S4BSmem& sm, const v4bh (&aa)[4],""")
rep_s4b("""    float dbacc = 0.f;
    #pragma unroll 1
    for (int kt = 0; kt < 8; ++kt) {
        const int buf = kt & 1, kc = 16 * kt;""", """    float dbacc = 0.f;
    float pdq = 0.f, pdk = 0.f, dAs = 0.f, Ag = 0.f;          // kra P-45 (TAIL)
    const short* gr_ = nullptr;
    if constexpr (TAIL) { Ag = expf(p.A_log[hh]); gr_ = p.g_raw + cb; }
    #pragma unroll 1
    for (int kt = 0; kt < 8; ++kt) {
        const int buf = kt & 1, kc = 16 * kt;""")
rep_s4b("""        const f4 gn = ld16f(g_ + rgn + kc);
        f4 dkp, dgK;""", """        const f4 gn = ld16f(g_ + rgn + kc);
        v4bh grw = v4bh{0, 0, 0, 0};                    // kra P-45: raw gate + dt_bias of the tile
        f4 dtb = f4{0.f, 0.f, 0.f, 0.f};
        if constexpr (TAIL) {
            grw = *reinterpret_cast<const v4bh*>(gr_ + rog + kc);
            dtb = ld16f(p.dt_bias + hh * kD + kc + 4 * q);
        }
        f4 dkp, dgK;""")
rep_s4b("""        dbacc += (dbp[0] + dbp[1]) + (dbp[2] + dbp[3]);
        if (rok) {
            *reinterpret_cast<f4*>(dq_ + ro + kc) = oq;
            *reinterpret_cast<f4*>(dk_ + ro + kc) = ok;
            *reinterpret_cast<f4*>(dg_ + ro + kc) = og;
        }
        lds_barrier();                             // table WAR: kt's reads done before kt+1 writes
    }
    db = xsum16(db) + xsum16(dbacc);
    if (q == 0 && rok) p.db[(size_t)(tok0 + row) * H + hh] = db;
}""", """        dbacc += (dbp[0] + dbp[1]) + (dbp[2] + dbp[3]);
        f4 sx = f4{0.f, 0.f, 0.f, 0.f};
        if constexpr (TAIL) {
            if (rok) {
                *reinterpret_cast<f4*>(dq_ + ro + kc) = oq;
                *reinterpret_cast<f4*>(dk_ + ro + kc) = ok;
            }
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                pdq += oq[e] * bf2f(qq[e]);                  // l2norm row dots (qq/kk = qn/kn tile)
                pdk += ok[e] * bf2f(kk[e]);
                float x = rok ? og[e] : 0.f;                 // rows past the chunk end contribute 0
                #pragma unroll
                for (int off = 1; off < 16; off <<= 1) {    // reverse inclusive suffix over the 16 rows
                    const float y = __shfl_down(x, off, 16);
                    x += (m + off < 16) ? y : 0.f;
                }
                sx[e] = x;
            }
            if (m == 0) {                                    // block total -> table pad row 64 + A
                #pragma unroll
                for (int e = 0; e < 4; ++e) sm.TA[(4 * e + q) * TS + 64 + A] = sx[e];
            }
        } else if (rok) {
            *reinterpret_cast<f4*>(dq_ + ro + kc) = oq;
            *reinterpret_cast<f4*>(dk_ + ro + kc) = ok;
            *reinterpret_cast<f4*>(dg_ + ro + kc) = og;
        }
        lds_barrier();                             // table WAR: kt's reads done before kt+1 writes
        if constexpr (TAIL) {
            // carry = suffix of the blocks below (higher rows), fixed order; then the gate backward
            // (FLA order: t1 = lb*dsig; bdg = dyg*t1; bdg *= A; dg_raw rounded to bf16 first)
            v4bh bo;
            f4 cv;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                float carry = 0.f;
                #pragma unroll
                for (int b = A + 1; b < 4; ++b) carry += sm.TA[(4 * e + q) * TS + 64 + b];
                const float suffix = sx[e] + carry;
                const float gg = bf2f(grw[e]) + dtb[e];
                const float sg = 1.f / (1.f + expf(-(Ag * gg)));
                float bdg = suffix * (p.lower_bound * (sg * (1.f - sg)));
                bdg *= Ag;
                dAs += rok ? bdg * gg : 0.f;
                bo[e] = bf_rne(bdg);
                cv[e] = rok ? bf2f(bo[e]) : 0.f;
            }
            if (rok) *reinterpret_cast<v4bh*>(p.dg_raw + cb + ro + kc) = bo;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {                    // ddt column partial over the wave's rows
                float v = cv[e];
                v += __shfl_xor(v, 1, 16); v += __shfl_xor(v, 2, 16);
                v += __shfl_xor(v, 4, 16); v += __shfl_xor(v, 8, 16);
                if (m == 0) p.ddt_part4[(((size_t)hh * kD + kc + 4 * q + e) * p.NT + i) * 4 + A] = v;
            }
        }
    }
    db = xsum16(db) + xsum16(dbacc);
    if constexpr (TAIL) {
        if (q == 0 && rok) p.db_out[(size_t)(tok0 + row) * H + hh] = db;
        {                                                    // this wave's dA partial
            float s = dAs;
            #pragma unroll
            for (int off = 1; off < 64; off <<= 1) s += __shfl_xor(s, off, 64);
            if (l == 0) p.dA_part4[((size_t)hh * p.NT + i) * 4 + A] = s;
        }
        // l2norm backward: dx = dy*rstd - (dy.y) y rstd  (row dot over the 4 lanes of the row)
        const float dqd = xsum16(pdq), dkd = xsum16(pdk);
        if (rok) {
            const float rq = p.q_rstd[(size_t)(tok0 + row) * H + hh];
            const float rk = p.k_rstd[(size_t)(tok0 + row) * H + hh];
            f4 dqv[8], dkv[8];
            v4bh qv[8], kv[8];
            #pragma unroll
            for (int kt = 0; kt < 8; ++kt) {
                dqv[kt] = *reinterpret_cast<const f4*>(dq_ + ro + 16 * kt);
                dkv[kt] = *reinterpret_cast<const f4*>(dk_ + ro + 16 * kt);
                qv[kt] = ld8(q_ + ro + 16 * kt);
                kv[kt] = ld8(k_ + ro + 16 * kt);
            }
            #pragma unroll
            for (int kt = 0; kt < 8; ++kt) {
                v4bh oq_, ok_;
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float yq = bf2f(qv[kt][e]), yk = bf2f(kv[kt][e]);
                    oq_[e] = bf_rne(dqv[kt][e] * rq - dqd * yq * rq);
                    ok_[e] = bf_rne(dkv[kt][e] * rk - dkd * yk * rk);
                }
                *reinterpret_cast<v4bh*>(p.dqo + cb + ro + 16 * kt) = oq_;
                *reinterpret_cast<v4bh*>(p.dko + cb + ro + 16 * kt) = ok_;
            }
        }
    } else {
        if (q == 0 && rok) p.db[(size_t)(tok0 + row) * H + hh] = db;
    }
}""")
# ---- wrapper kernel
rep("""template <bool VL>
__global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(1, 3)))
p3_wy_s4b_kernel(WyS4BParams p) {""", """template <bool VL, bool TAIL = false>
__global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(1, 3)))
p3_wy_s4b_kernel(WyS4BParams p) {""")
rep("""            case 0: s4b_tail<0, M>(p, sm, aa, db0, beta_t, tok0, cl); break;
            case 1: s4b_tail<1, M>(p, sm, aa, db0, beta_t, tok0, cl); break;
            case 2: s4b_tail<2, M>(p, sm, aa, db0, beta_t, tok0, cl); break;
            default: s4b_tail<3, M>(p, sm, aa, db0, beta_t, tok0, cl); break;""",
    """            case 0: s4b_tail<0, M, TAIL>(p, sm, aa, db0, beta_t, tok0, cl); break;
            case 1: s4b_tail<1, M, TAIL>(p, sm, aa, db0, beta_t, tok0, cl); break;
            case 2: s4b_tail<2, M, TAIL>(p, sm, aa, db0, beta_t, tok0, cl); break;
            default: s4b_tail<3, M, TAIL>(p, sm, aa, db0, beta_t, tok0, cl); break;""")
# ---- reduce kernel + fused launcher (appended after g2_wy_s4_split)
anchor = "// ================================================================ V1: l2n apply"
assert t.count(anchor) == 1
launcher = r'''// kra P-45: fixed-order reduce of the per-(chunk, wave) partials of the fused tail
__global__ void g2_s4bt_reduce_kernel(const float* __restrict__ dA_part4, const float* __restrict__ ddt_part4,
                                      float* dA_log, float* ddt_bias, int NT) {
    const int hh = blockIdx.x, tid = threadIdx.x;
    if (tid < kK) {
        const size_t base = ((size_t)hh * kK + tid) * NT * 4;
        float s = 0.f, c = 0.f;
        for (int k = 0; k < NT * 4; ++k) {
            const float y = ddt_part4[base + k] - c, tt = s + y;
            c = (tt - s) - y; s = tt;
        }
        ddt_bias[(size_t)hh * kK + tid] = bf2f(bf_rne(s));   // FLA: torch.sum on bf16 dg -> bf16
    } else if (tid == kK) {
        float s = 0.f;
        for (int k = 0; k < NT * 4; ++k) s += dA_part4[(size_t)hh * NT * 4 + k];
        dA_log[hh] = s;
    }
}

extern "C" int g2_wy_s4_split_t(const void* q_kgb, const void* k, const void* v_vb,
                                const void* v_new, const void* g, const void* beta,
                                const void* A, const void* h, const void* do_, const void* dh,
                                const void* dv, const void* dAqk, void* dq, void* dk,
                                void* dv2, void* db_part, void* dA, void* dwT,
                                float scale, int T, int H, int NT,
                                const void* q_row, const void* v_row, const int* ctok,
                                const int* clen, const void* do_row, const void* dv2c,
                                const void* vnc, const void* q_rstd, const void* k_rstd,
                                const void* g_raw, const void* A_log, const void* dt_bias,
                                void* dqo, void* dko, void* dg_raw, void* db_out, void* dA_log,
                                void* ddt_bias, void* dA_part4, void* ddt_part4,
                                double lower_bound, hipStream_t s) {
    if (T % kBT || NT != T / kBT || ((ctok == nullptr) != (clen == nullptr))) return -1;
    {
        WyS4AParams pa{(const bf16*)q_kgb, (const bf16*)v_vb, (const bf16*)A, (const bf16*)h,
                       (const bf16*)dv, (const float*)beta, (const bf16*)v_row,
                       (bf16*)dv2, (bf16*)dwT, (float*)dA, (float*)db_part, T, H, NT,
                       ctok, clen, (const bf16*)dv2c};
        int afs_maxh = 12;
        if (const char* e = getenv("KDA_G2_S4A_AFS_MAXH")) afs_maxh = atoi(e);
        const bool afs = H <= afs_maxh;
        if (ctok) {
            if (afs) hipLaunchKernelGGL((p3_wy_s4a_kernel<true, true>), dim3(NT, H), dim3(256), 0, s, pa);
            else     hipLaunchKernelGGL((p3_wy_s4a_kernel<true, false>), dim3(NT, H), dim3(256), 0, s, pa);
        } else {
            if (afs) hipLaunchKernelGGL((p3_wy_s4a_kernel<false, true>), dim3(NT, H), dim3(256), 0, s, pa);
            else     hipLaunchKernelGGL((p3_wy_s4a_kernel<false, false>), dim3(NT, H), dim3(256), 0, s, pa);
        }
        const int rc = (int)hipGetLastError();
        if (rc) return rc;
    }
    {
        WyS4BParams pb{(const bf16*)q_row, (const bf16*)k, (const bf16*)A, (const bf16*)h,
                       (const bf16*)do_, (const bf16*)dh, (const bf16*)v_new,
                       (const float*)g, (const float*)beta, (const float*)dAqk,
                       (const bf16*)dwT, (const float*)dA, (const float*)db_part,
                       (float*)dq, (float*)dk, nullptr, nullptr, scale, T, H, NT,
                       ctok, clen, (const bf16*)do_row, (const bf16*)vnc,
                       (const float*)q_rstd, (const float*)k_rstd, (const short*)g_raw,
                       (const float*)A_log, (const float*)dt_bias, (short*)dqo, (short*)dko,
                       (short*)dg_raw, (float*)db_out, (float*)dA_part4, (float*)ddt_part4,
                       (float)lower_bound};
        if (ctok) hipLaunchKernelGGL((p3_wy_s4b_kernel<true, true>), dim3(NT, H), dim3(256), 0, s, pb);
        else hipLaunchKernelGGL((p3_wy_s4b_kernel<false, true>), dim3(NT, H), dim3(256), 0, s, pb);
        const int rc = (int)hipGetLastError();
        if (rc) return rc;
    }
    hipLaunchKernelGGL(g2_s4bt_reduce_kernel, dim3(H), dim3(256), 0, s, (const float*)dA_part4,
                       (const float*)ddt_part4, (float*)dA_log, (float*)ddt_bias, NT);
    return (int)hipGetLastError();
}

'''
t = t.replace(anchor, launcher + anchor)
P.write_text(t)

# ---------------------------------------------------------------- bindings
B = root / "csrc" / "g2" / "g2_bindings.cpp"
b = B.read_text()
decl = '''extern "C" int g2_wy_s4_split_t(const void* q_kgb, const void* k, const void* v_vb,
                                const void* v_new, const void* g, const void* beta,
                                const void* A, const void* h, const void* do_, const void* dh,
                                const void* dv, const void* dAqk, void* dq, void* dk,
                                void* dv2, void* db_part, void* dA, void* dwT,
                                float scale, int T, int H, int NT,
                                const void* q_row, const void* v_row, const int* ctok,
                                const int* clen, const void* do_row, const void* dv2c,
                                const void* vnc, const void* q_rstd, const void* k_rstd,
                                const void* g_raw, const void* A_log, const void* dt_bias,
                                void* dqo, void* dko, void* dg_raw, void* db_out, void* dA_log,
                                void* ddt_bias, void* dA_part4, void* ddt_part4,
                                double lower_bound, hipStream_t s);
'''
anchor = "// Implemented in g2_ops.hip.\n"
assert b.count(anchor) == 1
b = b.replace(anchor, anchor + decl)
fn = r'''
// kra P-45: s4a + (s4b with the tail fused) + the partials' reduce.  Returns
// (dq bf16, dk bf16, dv2, dg_raw bf16, db fp32, dA_log fp32 [H], ddt_bias fp32 [H*128]).
std::vector<torch::Tensor> wy_s4bt(torch::Tensor q_kgb, torch::Tensor k, torch::Tensor v_vb,
                                   torch::Tensor v_new, torch::Tensor g, torch::Tensor beta,
                                   torch::Tensor A, torch::Tensor h, torch::Tensor do_,
                                   torch::Tensor dh, torch::Tensor dv, torch::Tensor q_row,
                                   torch::Tensor v_row, torch::Tensor dAqk, int64_t heads,
                                   int64_t tokens, double scale, torch::Tensor q_rstd,
                                   torch::Tensor k_rstd, torch::Tensor g_raw, torch::Tensor A_log,
                                   torch::Tensor dt_bias, double lower_bound,
                                   c10::optional<torch::Tensor> ctok,
                                   c10::optional<torch::Tensor> clen,
                                   c10::optional<torch::Tensor> do_row,
                                   c10::optional<torch::Tensor> vn_c,
                                   c10::optional<torch::Tensor> dv2_c) {
    const bool has_vn = v_new.numel() > 0, has_dv = dv.numel() > 0;
    const bool has_vb = v_vb.numel() > 0, has_ando = do_.numel() > 0;
    TORCH_CHECK(has_vn || (vn_c.has_value() && vn_c->numel() > 0), "wy_s4bt: aN_vn or vn_c");
    TORCH_CHECK(has_dv || (dv2_c.has_value() && dv2_c->numel() > 0), "wy_s4bt: aN_dv2 or dv2_c");
    TORCH_CHECK(has_ando || (do_row.has_value() && do_row->numel() > 0), "wy_s4bt: aN_do or do_row");
    for (auto* t_ : {&q_rstd, &k_rstd, &A_log, &dt_bias}) { check_contig(*t_, "wy_s4bt fp32"); check_f32(*t_, "wy_s4bt fp32"); }
    check_contig(g_raw, "wy_s4bt g_raw"); check_bf16(g_raw, "wy_s4bt g_raw");
    const int64_t H = heads, T = tokens, NT = T / kBT, B = 1;
    check_T(T);
    const RowMap rm = row_map(ctok, clen, k.numel() / (H * kK), NT, k);
    const int64_t Tr = rm.ctok ? rm.rows : T;
    c10::hip::HIPGuard guard(k.get_device());
    auto fopts = k.options().dtype(torch::kFloat32);
    auto bopts = k.options();
    auto dq = torch::empty({B, Tr, H, kK}, fopts);
    auto dk = torch::empty({B, Tr, H, kK}, fopts);
    auto dv2 = torch::empty({B, Tr, H, kK}, bopts);
    auto dA_out = torch::empty({B, T, H, kBT}, fopts);
    auto dwT = torch::empty({B, NT, H, kK, kBT}, bopts);
    auto db_part = torch::empty({B, NT, H, 256}, fopts);
    auto dqo = torch::empty({B, Tr, H, kK}, bopts);
    auto dko = torch::empty({B, Tr, H, kK}, bopts);
    auto dg_raw = torch::empty({B, Tr, H, kK}, bopts);
    auto db_out = torch::empty({B, Tr, H}, fopts);
    auto dA_log = torch::empty({H}, fopts);
    auto ddt_bias = torch::empty({H * kK}, fopts);
    auto dA_part4 = torch::empty({H * NT * 4}, fopts);
    auto ddt_part4 = torch::empty({H * kK * NT * 4}, fopts);
    const int rc = g2_wy_s4_split_t(q_kgb.data_ptr(), k.data_ptr(), has_vb ? v_vb.data_ptr() : nullptr,
                                    has_vn ? v_new.data_ptr() : nullptr, g.data_ptr(), beta.data_ptr(),
                                    A.data_ptr(), h.data_ptr(), has_ando ? do_.data_ptr() : nullptr,
                                    dh.data_ptr(), has_dv ? dv.data_ptr() : nullptr, dAqk.data_ptr(),
                                    dq.data_ptr(), dk.data_ptr(), dv2.data_ptr(), db_part.data_ptr(),
                                    dA_out.data_ptr(), dwT.data_ptr(), (float)scale, (int)T, (int)H,
                                    (int)NT, q_row.data_ptr(), v_row.data_ptr(), rm.ctok, rm.clen,
                                    has_ando ? nullptr : do_row->data_ptr(),
                                    has_dv ? nullptr : dv2_c->data_ptr(),
                                    has_vn ? nullptr : vn_c->data_ptr(), q_rstd.data_ptr(),
                                    k_rstd.data_ptr(), g_raw.data_ptr(), A_log.data_ptr(),
                                    dt_bias.data_ptr(), dqo.data_ptr(), dko.data_ptr(),
                                    dg_raw.data_ptr(), db_out.data_ptr(), dA_log.data_ptr(),
                                    ddt_bias.data_ptr(), dA_part4.data_ptr(), ddt_part4.data_ptr(),
                                    lower_bound, cur_stream());
    TORCH_CHECK(rc == 0, "g2_wy_s4_split_t failed: ", rc);
    return {dqo, dko, dv2, dg_raw, db_out, dA_log, ddt_bias};
}

'''
anchor2 = "\n}  // namespace\n\n// ------------------------------------------------------------ F1: HIP head (no FLA)"
assert b.count(anchor2) == 1, b.count(anchor2)
b = b.replace(anchor2, fn + anchor2)
anchor3 = '    m.def("fwd_prep", &fwd_prep, "G2 R4: single-pass forward prep (KDA_G2_FI=4)");'
assert b.count(anchor3) == 1
b = b.replace(anchor3, anchor3 + '\n    m.def("wy_s4bt", &wy_s4bt, "kra P-45: s4a + s4b with the bwd tail fused");')
B.write_text(b)

# ---------------------------------------------------------------- python
O = root / "hip_kda" / "ops" / "g2_ops.py"
o = O.read_text()
o += r'''

def wy_s4bt(q_kgb, k_row, v_vb, v_new_n, g, beta, a, h_n, do_n, dh_n, dv_n, q_row, v_row, dAqk,
            heads, tokens, scale, q_rstd, k_rstd, g_raw, a_log, dt_bias, lower_bound, rowmap=None,
            do_row=None, vn_c=None, dv2_c=None):
    """kra P-45 (B-class): wy_s4_split with the bwd tail fused into s4b -> (dq, dk, dv2, dg_raw, db,
    dA_log, ddt_bias); scope: gate on, qk l2norm on, beta sigmoid off."""
    return _ext().wy_s4bt(q_kgb, k_row, v_vb, v_new_n, g, beta, a, h_n, do_n, dh_n, dv_n, q_row,
                          v_row, dAqk, int(heads), int(tokens), float(scale),
                          q_rstd.contiguous(), k_rstd.contiguous(), g_raw.contiguous(),
                          a_log.contiguous(), dt_bias.reshape(-1).contiguous(), float(lower_bound),
                          *_rowmap(rowmap)[:2],
                          None if do_row is None else do_row.reshape(-1).contiguous(),
                          None if vn_c is None else vn_c.reshape(-1).contiguous(),
                          None if dv2_c is None else dv2_c.reshape(-1).contiguous())
'''
O.write_text(o)

K = root / "hip_kda" / "ops" / "k3_g2.py"
k = K.read_text()
old = """            if p4 == "3":
                # W1: split A/B kernels, 2 CTA/CU (bitwise equal to variant 2)
                dq, dk, dv_grad, dg, db = g2_ops.wy_s4_split(*wy_args, dAqk, H, 64 * NT, scale,"""
new = """            s4bt = (os.environ.get("KDA_G2_S4BT", "0").strip() == "1" and p4 == "3" and hipi
                    and not ctx.precomputed_g and ctx.use_qk_l2norm_in_kernel
                    and not ctx.use_beta_sigmoid_in_kernel and g_raw.dtype == torch.bfloat16)
            if s4bt:   # kra P-45: the tail rides in s4b (B-class)
                dqo, dko, dv_grad, dg_raw, db, dA_log, ddt_bias = g2_ops.wy_s4bt(
                    *wy_args, dAqk, H, 64 * NT, scale, q_rstd, k_rstd, g_raw, a_log.float(),
                    dt_bias.float(), ctx.lower_bound, rowmap=lay, do_row=do if no_ando else None,
                    vn_c=vn_c, dv2_c=dv2_c)
                dA_log = dA_log.to(a_log.dtype)
                ddt_bias = ddt_bias.view_as(dt_bias).to(dt_bias.dtype)
                return (dqo.to(qn), dko.to(kn), dv_grad.to(v), dg_raw.to(g_raw), db.to(beta_raw),
                        dA_log, ddt_bias, None, None, None, None, None, None, None, None)
            if p4 == "3":
                # W1: split A/B kernels, 2 CTA/CU (bitwise equal to variant 2)
                dq, dk, dv_grad, dg, db = g2_ops.wy_s4_split(*wy_args, dAqk, H, 64 * NT, scale,"""
assert k.count(old) == 1, k.count(old)
k = k.replace(old, new)
K.write_text(k)
print("P-45 (s4b + tail fusion) APPLIED")
