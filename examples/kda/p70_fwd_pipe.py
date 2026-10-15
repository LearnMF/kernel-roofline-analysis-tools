"""kra P-70 probe (plan A, bitwise): the forward recurrence overlaps the tail of the chunk-parallel prep.

C-1 (results/kda-gfx936-20261015-cu-util): at TP8 the recurrence holds 48 of 80 CUs; prep work run on the other 32
CUs while it runs is hidden (up to ~0.4 R).  Schedule (one call of the new binding fwd_pipe, dense only):
    main stream : prep_a + prep_b for chunks [0, NT-k)           (all CUs)
                  event E
                  prep_a for [NT-k, NT), then prep_b for [NT-k, NT), each prep_b CTA ends with a release store of
                  flag[i*H + hh] = epoch
    side stream : waits E, runs the recurrence; before prefetching chunk nxt >= NT-k each wave spins on
                  flag[nxt*H + bh] (acquire, agent scope)
    main waits for the side stream.
prep CTAs are dispatched chunk-major (grid (H, n), hh = blockIdx.x) so chunks complete in consumption order.  No
deadlock: the producers never wait.  Every value is computed by the same kernel code in the same order ->
bit-identical.  Probe plumbing: host globals g2_prep_range_set / g2_fwd_pipe_set (single-threaded probe only).
    python3 p70_fwd_pipe.py <tree>"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
g2 = root / "csrc" / "g2"


def sub(s, old, new, cnt=1):
    assert s.count(old) == cnt, (old[:80], s.count(old))
    return s.replace(old, new)


# ---------------------------------------------------------------- prep: chunk range, chunk-major grid, flags
p = g2 / "g2_fwd_prep.cuh"
s = p.read_text()
s = sub(s, """    int kp = 0;                         // kra P-61e: Kn lane-permuted + Gn natural-k (H <= KPERM_MAX_H)
};""", """    int kp = 0;                         // kra P-61e: Kn lane-permuted + Gn natural-k (H <= KPERM_MAX_H)
    int c_off = 0, cm = 0;              // kra P-70: chunk offset, chunk-major grid (hh = blockIdx.x)
};""")
s = sub(s, """    int kp = 0;   // kra P-62: Un with the pm column order (as Wn) when H <= KPERM_MAX_H
};""", """    int kp = 0;   // kra P-62: Un with the pm column order (as Wn) when H <= KPERM_MAX_H
    int c_off = 0, cm = 0;              // kra P-70
    int* flags = nullptr; int epoch = 0;   // kra P-70: per-(chunk, head) release flags
};""")
s = sub(s, "    const int i = blockIdx.x, hh = blockIdx.y;",
        "    const int i = (p.cm ? (int)blockIdx.y : (int)blockIdx.x) + p.c_off, hh = p.cm ? (int)blockIdx.x : (int)blockIdx.y;"
        "   // kra P-70", cnt=2)
s = sub(s, """        default: b_post<3, M>(p, smem, Rg, i, hh, l, tok0, cl); break;
    }
    });
}""", """        default: b_post<3, M>(p, smem, Rg, i, hh, l, tok0, cl); break;
    }
    });
    if (p.flags != nullptr) {   // kra P-70: chunk (i, hh) complete -> release
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0)
            __hip_atomic_store(p.flags + (size_t)i * p.H + hh, p.epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
}""")
s = sub(s, """extern "C" int g2_fwd_prep(const void* q, const void* k, const void* v, const void* g_raw,""",
        """// kra P-70 probe plumbing: chunk range / grid order / flags of the next g2_fwd_prep call
static int g2_pp_off = 0, g2_pp_n = -1, g2_pp_cm = 0, g2_pp_epoch = 0;
static int* g2_pp_flags = nullptr;
extern "C" void g2_prep_range_set(int off, int n, int cm, int* flags, int epoch) {
    g2_pp_off = off; g2_pp_n = n; g2_pp_cm = cm; g2_pp_flags = flags; g2_pp_epoch = epoch;
}
extern "C" int g2_fwd_prep(const void* q, const void* k, const void* v, const void* g_raw,""")
s = sub(s, """    if (const char* e = getenv("G2_PREP_STOP")) pa.stop = atoi(e);""",
        """    if (const char* e = getenv("G2_PREP_STOP")) pa.stop = atoi(e);
    pa.c_off = g2_pp_off; pa.cm = g2_pp_cm;   // kra P-70""")
s = sub(s, """    const dim3 grid(NT, H), blk(256);""",
        """    const int nrun = g2_pp_n >= 0 ? g2_pp_n : NT;   // kra P-70
    if (nrun <= 0) return 0;
    const dim3 grid(g2_pp_cm ? H : nrun, g2_pp_cm ? nrun : H), blk(256);""")
s = sub(s, """    pb.kp = pa.kp;   // kra P-62
""", """    pb.kp = pa.kp;   // kra P-62
    pb.c_off = g2_pp_off; pb.cm = g2_pp_cm; pb.flags = g2_pp_flags; pb.epoch = g2_pp_epoch;   // kra P-70
""")
p.write_text(s)

# ---------------------------------------------------------------- recurrence: flag waits
f = g2 / "g2_fwd_v8.cuh"
s = f.read_text()
s = sub(s, """    const float* GnRaw;                                // kra P-52: raw G_last (GE=false instance)
""", """    const float* GnRaw;                                // kra P-52: raw G_last (GE=false instance)
    const int* flags = nullptr; int i_tail = 0, epoch = 0;   // kra P-70: wait flag[nxt] before chunk nxt >= i_tail
""")
head, rest = s.split("g2_fwd_h_v8o4_kernel(FwdOParams p) {", 1)
body, tail = rest.split("// kra P-49: g2_fwd_h_v8o4_kernel with NVH", 1)
body = sub(body, """        const int nxt = i + 1 < c1 ? i + 1 : i;
""", """        const int nxt = i + 1 < c1 ? i + 1 : i;
        if (p.flags != nullptr && nxt >= p.i_tail) {   // kra P-70: chunk nxt produced?
            const int* fl = p.flags + (size_t)nxt * p.H + bh;
            while (__hip_atomic_load(fl, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < p.epoch)
                __builtin_amdgcn_s_sleep(2);
        }
""")
s = head + "g2_fwd_h_v8o4_kernel(FwdOParams p) {" + body + "// kra P-49: g2_fwd_h_v8o4_kernel with NVH" + tail
s = sub(s, """extern "C" int g2_fwd_h_v8o(const void* Wn, const void* Kn, const void* Un, const void* Gn,""",
        """// kra P-70 probe plumbing
static const int* g2_fp_flags = nullptr;
static int g2_fp_tail = 0, g2_fp_epoch = 0;
extern "C" void g2_fwd_pipe_set(const int* flags, int i_tail, int epoch) {
    g2_fp_flags = flags; g2_fp_tail = i_tail; g2_fp_epoch = epoch;
}
extern "C" int g2_fwd_h_v8o(const void* Wn, const void* Kn, const void* Un, const void* Gn,""")
s = sub(s, """                 scale, H, T / kBT, seg, nseg, ctok, clen, (const float*)GnRaw};
""", """                 scale, H, T / kBT, seg, nseg, ctok, clen, (const float*)GnRaw};
    p.flags = g2_fp_flags; p.i_tail = g2_fp_tail; p.epoch = g2_fp_epoch;   // kra P-70
""")
f.write_text(s)

# ---------------------------------------------------------------- binding fwd_pipe
b = g2 / "g2_bindings.cpp"
s = b.read_text()
s = sub(s, "extern \"C\" int g2_exp2_inplace(float* x, long n, hipStream_t s);   // kra P-52",
        "extern \"C\" int g2_exp2_inplace(float* x, long n, hipStream_t s);   // kra P-52\n"
        "extern \"C\" void g2_prep_range_set(int off, int n, int cm, int* flags, int epoch);   // kra P-70\n"
        "extern \"C\" void g2_fwd_pipe_set(const int* flags, int i_tail, int epoch);   // kra P-70")
anchor = "std::vector<torch::Tensor> fwd_prep(torch::Tensor q, torch::Tensor k, torch::Tensor v,"
s = sub(s, anchor, """static int g2_pipe_cm() { const char* e = getenv("KDA_G2_PIPE_CM"); return e ? atoi(e) : 0; }
// kra P-70 probe: prep (head chunks) -> [prep tail with flags || recurrence on a side stream]; dense, forward
// o only.  Returns (o_bf16, Aqk, Akk, q_rstd, k_rstd).
std::vector<torch::Tensor> fwd_pipe(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g_raw,
                                    torch::Tensor A_log, torch::Tensor dt_bias, torch::Tensor beta,
                                    double lower_bound, double gscale, double scale, int64_t heads,
                                    int64_t chunks, int64_t k_tail) {
    check_contig(q, "pipe q"); check_contig(k, "pipe k"); check_contig(v, "pipe v");
    check_contig(g_raw, "pipe g"); check_contig(beta, "pipe beta");
    TORCH_CHECK(q.size(1) == chunks * 64 && k_tail >= 1 && k_tail < chunks, "fwd_pipe: dense, 1 <= k < NT");
    c10::hip::HIPGuard guard(q.get_device());
    const auto opt = q.options();
    const auto f32 = opt.dtype(torch::kFloat32);
    const int64_t Tr = q.size(1), H = heads, NT = chunks, T = NT * 64;
    auto Wn = torch::empty({1, H, NT, 4, 8, 4, 16, 4}, opt);
    auto Kn = torch::empty({1, H, NT, 8, 4, 4, 16, 4}, opt);
    auto Un = torch::empty({1, H, NT, 4, 8, 4, 16, 4}, opt);
    auto QgA = torch::empty({1, H, NT, 4, 8, 4, 16, 4}, opt);
    auto AqkA = torch::empty({1, H, NT, 4, 4, 4, 16, 4}, opt);
    auto Aqk = torch::empty({1, Tr, H, 64}, opt);
    auto Akk = torch::empty({1, Tr, H, 64}, opt);
    auto qr = torch::empty({1, Tr, H}, f32), kr = torch::empty({1, Tr, H}, f32);
    auto Gn = torch::empty({1, H, NT, 8, 4, 4}, f32);
    auto R = torch::empty({1, H, NT, 5120}, f32);
    auto Kb = torch::empty_like(q);
    auto flags = torch::zeros({H * NT}, opt.dtype(torch::kInt32));
    auto Ob = torch::empty({1, T, H, 128}, opt);
    hipStream_t main = cur_stream();
    static hipStream_t side = nullptr;
    if (side == nullptr) TORCH_CHECK(hipStreamCreateWithFlags(&side, hipStreamNonBlocking) == hipSuccess, "side stream");
    auto prep = [&](int off, int n, int* fl) {
        g2_prep_range_set(off, n, g2_pipe_cm(), fl, 1);   // chunk-major only on request (it slows prep)
        const int rc = g2_fwd_prep(q.data_ptr(), k.data_ptr(), v.data_ptr(), g_raw.data_ptr(), A_log.data_ptr(),
                                   dt_bias.data_ptr(), beta.data_ptr(), Wn.data_ptr(), Kn.data_ptr(), Un.data_ptr(),
                                   QgA.data_ptr(), AqkA.data_ptr(), Aqk.data_ptr(), Akk.data_ptr(), qr.data_ptr(),
                                   kr.data_ptr(), Gn.data_ptr(), R.data_ptr(), Kb.data_ptr(), lower_bound, gscale,
                                   1, scale, (int)H, (int)NT, nullptr, nullptr, main);
        g2_prep_range_set(0, -1, 0, nullptr, 0);
        TORCH_CHECK(rc == 0, "fwd_pipe prep failed: ", rc);
    };
    const int i_tail = (int)(NT - k_tail);
    prep(0, i_tail, nullptr);
    hipEvent_t e0, e1;
    hipEventCreateWithFlags(&e0, hipEventDisableTiming);
    hipEventCreateWithFlags(&e1, hipEventDisableTiming);
    hipEventRecord(e0, main);
    hipStreamWaitEvent(side, e0, 0);
    g2_fwd_pipe_set(flags.data_ptr<int>(), i_tail, 1);
    const int rc = g2_fwd_h_v8o(Wn.data_ptr(), Kn.data_ptr(), Un.data_ptr(), Gn.data_ptr(), QgA.data_ptr(),
                                AqkA.data_ptr(), nullptr, Ob.data_ptr(), nullptr, nullptr, nullptr, (float)scale,
                                1, (int)T, (int)H, nullptr, 1, nullptr, nullptr, nullptr, side);
    g2_fwd_pipe_set(nullptr, 0, 0);
    TORCH_CHECK(rc == 0, "fwd_pipe recurrence failed: ", rc);
    prep(i_tail, (int)k_tail, flags.data_ptr<int>());   // the tail on the main stream, concurrently
    hipEventRecord(e1, side);
    hipStreamWaitEvent(main, e1, 0);
    hipEventDestroy(e0); hipEventDestroy(e1);
    return {Ob, Aqk, Akk, qr, kr};
}

""" + anchor)
s = sub(s, '    m.def("fwd_prep", &fwd_prep, "G2 R4: single-pass forward prep (KDA_G2_FI=4)");',
        '    m.def("fwd_prep", &fwd_prep, "G2 R4: single-pass forward prep (KDA_G2_FI=4)");\n'
        '    m.def("fwd_pipe", &fwd_pipe, "G2: prep tail || recurrence (kra P-70 probe)");')
b.write_text(s)

# ---------------------------------------------------------------- python wiring (env KDA_G2_PIPE=1)
k = root / "hip_kda" / "ops" / "k3_g2.py"
s = k.read_text()
old = """            Wn, Kn, Un, QgA, AqkA, Aqk, Akk, q_rstd, k_rstd, Gn = g2_ops.fwd_prep(
                q, k, v, g_raw, a_log.float() if gate else None, dt_bias.float() if gate else None,
                beta, lower_bound if gate else 0.0, RCP_LN2, gate, scale, H, NT, rowmap=lay)
            obf = os.environ.get("KDA_G2_OBF", "1").strip() != "0"
            keep = keep_state(ctx)
            outs = g2_ops.fwd_h_o(Wn, Kn, Un, Gn, QgA, AqkA, scale, emit_bn=keep, o_bf16=obf,
                                  seg=seg, rowmap=lay, emit_vn=keep, emit_anv=keep)
            o = outs[0]
            kept = (outs[2], outs[1], outs[3]) if keep else (None, None, None)   # bN_h, Vn, aN_vn
            del Wn, Kn, Un, Gn, QgA, AqkA, outs"""
new = """            obf = os.environ.get("KDA_G2_OBF", "1").strip() != "0"
            keep = keep_state(ctx)
            ktail = int(os.environ.get("KDA_G2_PIPE_K", str(max(1, round(0.18 * NT)))))
            if (os.environ.get("KDA_G2_PIPE", "0") == "1" and lay is None and seg is None and gate and obf
                    and not keep and 1 <= ktail < NT):   # kra P-70 probe: prep tail || recurrence
                o, Aqk, Akk, q_rstd, k_rstd = g2_ops._ext().fwd_pipe(
                    q.contiguous(), k.contiguous(), v.contiguous(), g_raw.contiguous(), a_log.float().contiguous(),
                    dt_bias.float().contiguous(), beta.contiguous(), float(lower_bound), float(RCP_LN2), float(scale),
                    int(H), int(NT), int(ktail))
                kept = (None, None, None)
            else:
                Wn, Kn, Un, QgA, AqkA, Aqk, Akk, q_rstd, k_rstd, Gn = g2_ops.fwd_prep(
                    q, k, v, g_raw, a_log.float() if gate else None, dt_bias.float() if gate else None,
                    beta, lower_bound if gate else 0.0, RCP_LN2, gate, scale, H, NT, rowmap=lay)
                outs = g2_ops.fwd_h_o(Wn, Kn, Un, Gn, QgA, AqkA, scale, emit_bn=keep, o_bf16=obf,
                                      seg=seg, rowmap=lay, emit_vn=keep, emit_anv=keep)
                o = outs[0]
                kept = (outs[2], outs[1], outs[3]) if keep else (None, None, None)   # bN_h, Vn, aN_vn
                del Wn, Kn, Un, Gn, QgA, AqkA, outs"""
s = sub(s, old, new)
k.write_text(s)
print("P-70 APPLIED")
