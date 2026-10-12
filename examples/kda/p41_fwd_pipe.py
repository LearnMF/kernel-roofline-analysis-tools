"""kra P-41 (round 13, Phase A: temporal pipeline, A-class = bit-identical): the forward's parallel
prep (prep_a + prep_b, chunk-parallel) and its sequential recurrence (fwd_h_v8o4, H x 4 CTAs) run
CONCURRENTLY on two streams over chunk ranges: recurrence(range j) overlaps prep(range j+1).

Measured motivation (overlap_probe.py, independent data on two streams): co-running hides 41% of the
recurrence at 16K/TP8, 32% at 32K/TP4, 55% at 64K/TP8 (0% at TP2, where the recurrence fills the
GPU).  The recurrence resumes each range from the previous range's fp32 register state (an opaque
per-thread image, exact) -> every output is bit-identical to the one-shot forward.

  kernels  g2_fwd_h_v8o4_kernel: chunk range [cb, ce) + h_in / h_out (dense only)
           g2_fwd_prep_a/b_kernel: chunk offset c0 (grid (n, H))
  launch   g2_fwd_prep_rng, g2_fwd_h_v8o_rng
  bind     _G2.fwd_prep_rng (writes caller-allocated outputs), _G2.fwd_h_o_rng
  python   g2_ops.fwd_pipe(...); k3_g2 forward uses it when KDA_G2_FPIPE=<pieces> (>1), dense,
           o in bf16, no keep-state.
Usage: python3 p41_fwd_pipe.py <tree>"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
G = root / "csrc" / "g2"


def patch(path, pairs):
    t = path.read_text()
    for old, new, cnt in pairs:
        assert t.count(old) == cnt, (path.name, old[:80], t.count(old))
        t = t.replace(old, new)
    path.write_text(t)


# ---------------------------------------------------------------- recurrence kernel + launcher
patch(G / "g2_fwd_v8.cuh", [
    ("    const int* ctok; const int* clen;\n};",
     "    const int* ctok; const int* clen;\n"
     "    // kra P-41: temporal pipeline (dense only) -- chunks [cb, ce) when ce > 0; resume from h_in\n"
     "    // (this kernel's own per-thread fp32 state image) and leave the range's final state in h_out\n"
     "    int cb, ce; const float* h_in; float* h_out;\n};", 1),
    ("    const int c0 = p.seg ? p.seg[sg] : 0, c1 = p.seg ? p.seg[sg + 1] : p.NT;\n    if (c0 >= c1) return;\n",
     "    int c0 = p.seg ? p.seg[sg] : 0, c1 = p.seg ? p.seg[sg + 1] : p.NT;\n"
     "    if (p.ce > 0) { c0 = p.cb; c1 = p.ce; }            // kra P-41: chunk range (dense)\n"
     "    if (c0 >= c1) return;\n", 1),
    ("    ht1[0] = f4{0.f,0.f,0.f,0.f}; ht1[1] = ht1[0];\n",
     "    ht1[0] = f4{0.f,0.f,0.f,0.f}; ht1[1] = ht1[0];\n"
     "    const size_t hio = ((size_t)(bh * (kK / kBV) + vs) * 256 + tid) * 16;   // kra P-41\n"
     "    if (p.h_in) {\n"
     "        const f4* src = reinterpret_cast<const f4*>(p.h_in + hio);\n"
     "        ht0[0] = src[0]; ht0[1] = src[1]; ht1[0] = src[2]; ht1[1] = src[3];\n"
     "    }\n", 1),
    ("        write_h(kb0, ht0);\n        write_h(kb1, ht1);\n    }\n}\n",
     "        write_h(kb0, ht0);\n        write_h(kb1, ht1);\n    }\n"
     "    if (p.h_out) {                                      // kra P-41: final state of the range\n"
     "        f4* dst = reinterpret_cast<f4*>(p.h_out + hio);\n"
     "        dst[0] = ht0[0]; dst[1] = ht0[1]; dst[2] = ht1[0]; dst[3] = ht1[1];\n"
     "    }\n}\n", 1),
])
t = (G / "g2_fwd_v8.cuh").read_text()
t += r'''
// kra P-41: one chunk range of the forward recurrence (dense, o in bf16 only = MODE 1).
extern "C" int g2_fwd_h_v8o_rng(const void* Wn, const void* Kn, const void* Un, const void* Gn,
                                const void* QgA, const void* AqkA, void* Ob, float scale, int T,
                                int H, int cb, int ce, const float* h_in, float* h_out,
                                hipStream_t s) {
    using namespace g2;
    if (T % kBT || cb < 0 || ce > T / kBT || cb >= ce || Ob == nullptr) return -1;
    FwdOParams p{(const bf16*)Wn, (const bf16*)Kn, (const bf16*)Un, (const float*)Gn,
                 (const bf16*)QgA, (const bf16*)AqkA, nullptr, (bf16*)Ob, nullptr, nullptr, nullptr,
                 scale, H, T / kBT, nullptr, 1, nullptr, nullptr, cb, ce, h_in, h_out};
    const dim3 grid(kK / kBV, (unsigned)H);
    const bool throttle = 4L * H >= g2_env_long("KDA_G2_V8O4_THROTTLE_MIN_CTAS", 224);
    const size_t dyn = throttle ? g2_lds_topup(g2_fwd_h_v8o4_kernel<false, 1>, 24 * 1024) : 0;
    hipLaunchKernelGGL((g2_fwd_h_v8o4_kernel<false, 1>), grid, dim3(256), dyn, s, p);
    return (int)hipGetLastError();
}
'''
(G / "g2_fwd_v8.cuh").write_text(t)

# ---------------------------------------------------------------- prep kernels + launcher
patch(G / "g2_fwd_prep.cuh", [
    ("    int stop;                           // phase timing (G2_PREP_STOP): 0 = full\n",
     "    int stop;                           // phase timing (G2_PREP_STOP): 0 = full\n"
     "    int c0;                             // kra P-41: chunk offset of a range launch\n", 1),
    ("    float scale;\n    int H, NT;\n};",
     "    float scale;\n    int H, NT;\n    int c0;                             // kra P-41\n};", 1),
    ("    const int i = blockIdx.x, hh = blockIdx.y;",
     "    const int i = blockIdx.x + p.c0, hh = blockIdx.y;   // kra P-41: + range offset", 2),
])
t = (G / "g2_fwd_prep.cuh").read_text()
t += r'''
// kra P-41: prep_a + prep_b over chunks [c0, c0 + n) into caller-allocated outputs.
extern "C" int g2_fwd_prep_rng(const void* q, const void* k, const void* v, const void* g_raw,
                               const void* A_log, const void* dt_bias, const void* beta,
                               void* Wn, void* Kn, void* Un, void* QgA, void* AqkA, void* Aqk,
                               void* Akk, void* q_rstd, void* k_rstd, void* Gn, void* R, void* Kb,
                               double lower_bound, double gscale, int gate, double scale, int H,
                               int NT, int c0, int n, hipStream_t s) {
    if (NT <= 0 || c0 < 0 || n <= 0 || c0 + n > NT) return -1;
    g2::r4::PrepAParams pa{(const short*)q, (const short*)k, (const short*)v, (short*)Wn,
                           (short*)Un, (short*)AqkA, (short*)Aqk, (short*)Akk, (float)scale,
                           g_raw, (const float*)A_log, (const float*)dt_bias, (const float*)beta,
                           (short*)QgA, (short*)Kn, (short*)Kb, (float*)R, (float*)q_rstd,
                           (float*)k_rstd, (float*)Gn, nullptr, nullptr, (float)lower_bound,
                           (float)gscale, H, NT, 0, c0};
    const dim3 grid(n, H), blk(256);
    if (gate) hipLaunchKernelGGL((g2::r4::g2_fwd_prep_a_kernel<1, false>), grid, blk, 0, s, pa);
    else      hipLaunchKernelGGL((g2::r4::g2_fwd_prep_a_kernel<0, false>), grid, blk, 0, s, pa);
    int rc = (int)hipGetLastError();
    if (rc) return rc;
    g2::r4::PrepBParams pb{(const short*)Kb, (const short*)v, (const float*)beta, (const float*)R,
                           (short*)Wn, (short*)Un, (short*)AqkA, (short*)Aqk, (short*)Akk,
                           nullptr, nullptr, (float)scale, H, NT, c0};
    hipLaunchKernelGGL(g2::r4::g2_fwd_prep_b_kernel<false>, grid, blk, 0, s, pb);
    return (int)hipGetLastError();
}
'''
(G / "g2_fwd_prep.cuh").write_text(t)

# ---------------------------------------------------------------- bindings
B = G / "g2_bindings.cpp"
t = B.read_text()
decl = '''extern "C" int g2_fwd_prep_rng(const void* q, const void* k, const void* v, const void* g_raw,
                               const void* A_log, const void* dt_bias, const void* beta,
                               void* Wn, void* Kn, void* Un, void* QgA, void* AqkA, void* Aqk,
                               void* Akk, void* q_rstd, void* k_rstd, void* Gn, void* R, void* Kb,
                               double lower_bound, double gscale, int gate, double scale, int H,
                               int NT, int c0, int n, hipStream_t s);
extern "C" int g2_fwd_h_v8o_rng(const void* Wn, const void* Kn, const void* Un, const void* Gn,
                                const void* QgA, const void* AqkA, void* Ob, float scale, int T,
                                int H, int cb, int ce, const float* h_in, float* h_out,
                                hipStream_t s);
'''
anchor = "// Implemented in g2_ops.hip.\n"
assert t.count(anchor) == 1
t = t.replace(anchor, anchor + decl)
fns = r'''
// kra P-41: range launches for the two-stream forward pipeline (caller allocates every output).
void fwd_prep_rng(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g_raw,
                  c10::optional<torch::Tensor> A_log, c10::optional<torch::Tensor> dt_bias,
                  torch::Tensor beta, std::vector<torch::Tensor> outs, double lower_bound,
                  double gscale, bool gate, double scale, int64_t heads, int64_t chunks,
                  int64_t c0, int64_t n) {
    TORCH_CHECK(outs.size() == 12, "fwd_prep_rng: 12 outputs (Wn Kn Un QgA AqkA Aqk Akk qr kr Gn R Kb)");
    TORCH_CHECK(q.size(1) == chunks * 64, "fwd_prep_rng: dense only");
    for (auto& o : outs) check_contig(o, "fwd_prep_rng out");
    c10::hip::HIPGuard guard(q.get_device());
    const int rc = g2_fwd_prep_rng(q.data_ptr(), k.data_ptr(), v.data_ptr(), g_raw.data_ptr(),
                                   gate ? A_log->data_ptr() : nullptr, gate ? dt_bias->data_ptr() : nullptr,
                                   beta.data_ptr(), outs[0].data_ptr(), outs[1].data_ptr(),
                                   outs[2].data_ptr(), outs[3].data_ptr(), outs[4].data_ptr(),
                                   outs[5].data_ptr(), outs[6].data_ptr(), outs[7].data_ptr(),
                                   outs[8].data_ptr(), outs[9].data_ptr(), outs[10].data_ptr(),
                                   outs[11].data_ptr(), lower_bound, gscale, gate ? 1 : 0, scale,
                                   (int)heads, (int)chunks, (int)c0, (int)n, cur_stream());
    TORCH_CHECK(rc == 0, "g2_fwd_prep_rng failed: ", rc);
}

void fwd_h_o_rng(torch::Tensor Wn, torch::Tensor Kn, torch::Tensor Un, torch::Tensor Gn,
                 torch::Tensor QgA, torch::Tensor AqkA, torch::Tensor Ob, double scale,
                 int64_t cb, int64_t ce, c10::optional<torch::Tensor> h_in,
                 c10::optional<torch::Tensor> h_out) {
    const int64_t H = Wn.size(1), NT = Wn.size(2);
    check_contig(Ob, "fwd_h_o_rng Ob"); check_bf16(Ob, "fwd_h_o_rng Ob");
    for (auto* t : {&h_in, &h_out})
        if (t->has_value()) {
            check_f32(**t, "fwd_h_o_rng state"); check_contig(**t, "fwd_h_o_rng state");
            TORCH_CHECK((*t)->numel() == H * 4 * 256 * 16, "fwd_h_o_rng: state size");
        }
    c10::hip::HIPGuard guard(Wn.get_device());
    const int rc = g2_fwd_h_v8o_rng(Wn.data_ptr(), Kn.data_ptr(), Un.data_ptr(), Gn.data_ptr(),
                                    QgA.data_ptr(), AqkA.data_ptr(), Ob.data_ptr(), (float)scale,
                                    (int)(NT * 64), (int)H, (int)cb, (int)ce,
                                    h_in.has_value() ? h_in->data_ptr<float>() : nullptr,
                                    h_out.has_value() ? h_out->data_ptr<float>() : nullptr,
                                    cur_stream());
    TORCH_CHECK(rc == 0, "g2_fwd_h_v8o_rng failed: ", rc);
}

'''
anchor2 = "std::vector<torch::Tensor> do_dual(torch::Tensor x, int64_t heads, int64_t chunks) {"
assert t.count(anchor2) == 1
t = t.replace(anchor2, fns + anchor2)
anchor3 = '    m.def("fwd_prep", &fwd_prep, "G2 R4: single-pass forward prep (KDA_G2_FI=4)");'
assert t.count(anchor3) == 1
t = t.replace(anchor3, anchor3 + '\n    m.def("fwd_prep_rng", &fwd_prep_rng, "kra P-41: prep over a chunk range");'
                               '\n    m.def("fwd_h_o_rng", &fwd_h_o_rng, "kra P-41: forward recurrence over a chunk range");')
B.write_text(t)

# ---------------------------------------------------------------- python: g2_ops.fwd_pipe + k3_g2 route
O = root / "hip_kda" / "ops" / "g2_ops.py"
t = O.read_text()
t += r'''

_PIPE_STREAMS = {}


def fwd_pipe(q, k, v, g_raw, a_log, dt_bias, beta, lower_bound, gscale, gate, scale, heads,
             chunks, pieces):
    """kra P-41: the forward as a two-stream temporal pipeline -- prep over chunk range j+1 runs
    while the recurrence processes range j (state handed over in fp32, bit-identical).  Dense,
    bf16 o.  Returns (o, Aqk, Akk, q_rstd, k_rstd)."""
    import torch  # noqa: PLC0415
    ext = _ext()
    dev = q.device
    opt = dict(device=dev, dtype=q.dtype)
    f32 = dict(device=dev, dtype=torch.float32)
    H, NT, Tr = int(heads), int(chunks), q.size(1)
    outs = [torch.empty((1, H, NT, 4, 8, 4, 16, 4), **opt), torch.empty((1, H, NT, 8, 4, 4, 16, 4), **opt),
            torch.empty((1, H, NT, 4, 8, 4, 16, 4), **opt), torch.empty((1, H, NT, 4, 8, 4, 16, 4), **opt),
            torch.empty((1, H, NT, 4, 4, 4, 16, 4), **opt), torch.empty((1, Tr, H, 64), **opt),
            torch.empty((1, Tr, H, 64), **opt), torch.empty((1, Tr, H), **f32), torch.empty((1, Tr, H), **f32),
            torch.empty((1, H, NT, 8, 4, 4), **f32), torch.empty((1, H, NT, 5120), **f32), torch.empty_like(q)]
    Wn, Kn, Un, QgA, AqkA, Aqk, Akk, qr, kr, Gn = outs[:10]
    o = torch.empty((1, Tr, H, 128), **opt)
    p = max(1, min(int(pieces), NT))
    cuts = [NT * j // p for j in range(p + 1)]
    hs = [torch.empty(H * 4 * 256 * 16, **f32) for _ in range(p - 1)]
    key = dev.index if dev.index is not None else torch.cuda.current_device()
    if key not in _PIPE_STREAMS:
        _PIPE_STREAMS[key] = (torch.cuda.Stream(dev), torch.cuda.Stream(dev))
    sp, sr = _PIPE_STREAMS[key]
    main = torch.cuda.current_stream(dev)
    sp.wait_stream(main)
    sr.wait_stream(main)
    al = a_log if gate else None
    dt = dt_bias if gate else None
    evs = []
    for j in range(p):
        with torch.cuda.stream(sp):
            ext.fwd_prep_rng(q, k, v, g_raw, al, dt, beta, outs, float(lower_bound or 0.0),
                             float(gscale), bool(gate), float(scale), H, NT, cuts[j], cuts[j + 1] - cuts[j])
            e = torch.cuda.Event()
            e.record(sp)
        evs.append(e)
    for j in range(p):
        sr.wait_event(evs[j])
        with torch.cuda.stream(sr):
            ext.fwd_h_o_rng(Wn, Kn, Un, Gn, QgA, AqkA, o, float(scale), cuts[j], cuts[j + 1],
                            hs[j - 1] if j > 0 else None, hs[j] if j < p - 1 else None)
    main.wait_stream(sp)
    main.wait_stream(sr)
    return o, Aqk, Akk, qr, kr
'''
O.write_text(t)

K = root / "hip_kda" / "ops" / "k3_g2.py"
t = K.read_text()
old = """            gate = not precomputed_g
            Wn, Kn, Un, QgA, AqkA, Aqk, Akk, q_rstd, k_rstd, Gn = g2_ops.fwd_prep("""
new = """            gate = not precomputed_g
            pipe = int(os.environ.get("KDA_G2_FPIPE", "1").strip() or 1)   # kra P-41
            if (pipe > 1 and lay is None and seg is None and not keep_state(ctx)
                    and os.environ.get("KDA_G2_OBF", "1").strip() != "0"):
                o, Aqk, Akk, q_rstd, k_rstd = g2_ops.fwd_pipe(
                    q, k, v, g_raw, a_log.float() if gate else None, dt_bias.float() if gate else None,
                    beta, lower_bound if gate else 0.0, RCP_LN2, gate, scale, H, NT, pipe)
                ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g_raw, beta_raw, beta,
                                      a_log, dt_bias, Aqk, Akk, None, None, None)
                ctx.scale = scale
                ctx.lower_bound = lower_bound
                ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
                ctx.use_beta_sigmoid_in_kernel = use_beta_sigmoid_in_kernel
                ctx.allow_neg_eigval = allow_neg_eigval
                ctx.precomputed_g = precomputed_g
                ctx.seg = seg
                ctx.lay = lay
                return o.type_as(q)
            Wn, Kn, Un, QgA, AqkA, Aqk, Akk, q_rstd, k_rstd, Gn = g2_ops.fwd_prep("""
assert t.count(old) == 1, t.count(old)
t = t.replace(old, new)
K.write_text(t)
print("P-41 (forward temporal pipeline) APPLIED")
