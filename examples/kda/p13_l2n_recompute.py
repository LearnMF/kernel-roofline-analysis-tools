"""kra P-13 (backward fusion step 1): drop the g2_l2n_apply launch -- its only consumers on
the production backward route (wu, wy_s4b, bwd_tail) read the RAW q/k plus the forward's
saved q_rstd/k_rstd and recompute qn = bf_rne(bf2f(x) * rstd[row]) at load, the exact
expression of g2_l2n_apply_kernel (bit-identical).  Old extern "C" entry points are kept
(nullptr rstd); new ones: g2_wu2, g2_wy_s4_split2, g2_bwd_tail2.  KDA_G2_L2R=0 restores
the l2n_apply launch.
Usage: python3 p13_l2n_recompute.py <hip_kda tree root> [--kernels-only]"""
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
KONLY = "--kernels-only" in sys.argv
K = ROOT if KONLY else ROOT / "csrc" / "g2"


def sub(path, old, new, count=1, regex=False):
    t = path.read_text()
    if regex:
        n = len(re.findall(old, t))
        assert n == count, (path.name, old[:70], n)
        t = re.sub(old, new, t)
    else:
        n = t.count(old)
        assert n == count, (path.name, old[:70], n)
        t = t.replace(old, new)
    path.write_text(t)


# ---------------------------------------------------------------- common helper
BF2F = "__device__ __forceinline__ float bf2f(short s) { return __uint_as_float(((uint32_t)(unsigned short)s) << 16); }\n"
sub(K / "g2_common.h", BF2F,
    BF2F + """// kra P-13: l2-normalize a raw bf16 quad exactly like g2_l2n_apply_kernel
__device__ __forceinline__ v4bh l2n4(v4bh x, float r) {
    v4bh o;
    #pragma unroll
    for (int e = 0; e < 4; ++e) o[e] = bf_rne(bf2f(x[e]) * r);
    return o;
}
""")

# ---------------------------------------------------------------- wu
W = K / "g2_wu.cuh"
sub(W, r"(    int stop;      // T2 staged timing \(G2_WU_STOP\)[^\n]*\n[^\n]*\n)",
    r"\1    const float* k_rstd; const float* q_rstd;   // kra P-13: raw k/q + rstd (null = qn/kn given)\n",
    regex=True)
sub(W, r"const v4bh kv = g2_row_ok<M>\(j, cl\) \? \*reinterpret_cast<const v4bh\*>\(p\.k \+ so\)\s*: v4bh\{0, 0, 0, 0\};",
    "v4bh kv = g2_row_ok<M>(j, cl) ? *reinterpret_cast<const v4bh*>(p.k + so) : v4bh{0, 0, 0, 0};\n"
    "            if (p.k_rstd != nullptr && g2_row_ok<M>(j, cl)) kv = l2n4(kv, p.k_rstd[so >> 7]);   // kra P-13",
    regex=True)
sub(W, r"const v4bh qv = g2_row_ok<M>\(j, cl\) \? \*reinterpret_cast<const v4bh\*>\(p\.q \+ so\)\s*: v4bh\{0, 0, 0, 0\};",
    "v4bh qv = g2_row_ok<M>(j, cl) ? *reinterpret_cast<const v4bh*>(p.q + so) : v4bh{0, 0, 0, 0};\n"
    "                if (p.q_rstd != nullptr && g2_row_ok<M>(j, cl)) qv = l2n4(qv, p.q_rstd[so >> 7]);   // kra P-13",
    regex=True)
sub(W, r"const v4bh kv = g2_row_ok<M>\(j, cl\)\s*\? \*reinterpret_cast<const v4bh\*>\(p\.k \+ sbase \+ \(size_t\)j \* rstride \+ vq\)\s*: v4bh\{0, 0, 0, 0\};",
    "v4bh kv = g2_row_ok<M>(j, cl)\n"
    "                        ? *reinterpret_cast<const v4bh*>(p.k + sbase + (size_t)j * rstride + vq)\n"
    "                        : v4bh{0, 0, 0, 0};\n"
    "                    if (p.k_rstd != nullptr && g2_row_ok<M>(j, cl))   // kra P-13\n"
    "                        kv = l2n4(kv, p.k_rstd[(sbase + (size_t)j * rstride + vq) >> 7]);",
    regex=True)
sub(W, """extern "C" int g2_wu(const void* k, const void* v, const void* q, const void* beta,
                     const void* A, const void* g,
                     void* Wn, void* Wb, void* Un, void* KgA, void* Kn, void* Qn,
                     void* kgB, void* vB,
                     int B, int T, int H, const int* ctok, const int* clen, hipStream_t s) {""",
    """extern "C" int g2_wu2(const void* k, const void* v, const void* q, const void* beta,
                      const void* A, const void* g,
                      void* Wn, void* Wb, void* Un, void* KgA, void* Kn, void* Qn,
                      void* kgB, void* vB,
                      int B, int T, int H, const int* ctok, const int* clen,
                      const void* q_rstd, const void* k_rstd, hipStream_t s) {""")
sub(W, """                   H, T / 64, ctok, clen, stop};
    if (ctok)
        hipLaunchKernelGGL(g2::g2_wu_kernel<true>""",
    """                   H, T / 64, ctok, clen, stop};
    p.k_rstd = (const float*)k_rstd; p.q_rstd = (const float*)q_rstd;   // kra P-13
    if (ctok)
        hipLaunchKernelGGL(g2::g2_wu_kernel<true>""")
t = W.read_text()
i = t.index('extern "C" int g2_wu2(')
j = t.index("\n}\n", i) + 3
t = t[:j] + """
extern "C" int g2_wu(const void* k, const void* v, const void* q, const void* beta,
                     const void* A, const void* g,
                     void* Wn, void* Wb, void* Un, void* KgA, void* Kn, void* Qn,
                     void* kgB, void* vB,
                     int B, int T, int H, const int* ctok, const int* clen, hipStream_t s) {
    return g2_wu2(k, v, q, beta, A, g, Wn, Wb, Un, KgA, Kn, Qn, kgB, vB, B, T, H, ctok, clen,
                  nullptr, nullptr, s);
}
""" + t[j:]
W.write_text(t)

# ---------------------------------------------------------------- wy_s4b
Y = K / "g2_wy.cuh"
t = Y.read_text()
s0 = t.index("struct WyS4BParams {")
s1 = t.index("};", s0)
blk = t[s0:s1]
assert blk.count("const int* ctok; const int* clen;") == 1
blk = blk.replace("const int* ctok; const int* clen;",
                  "const int* ctok; const int* clen;\n    const float* q_rstd; const float* k_rstd;   // kra P-13: raw q/k + rstd (null = qn/kn given)", 1)
t = t[:s0] + blk + t[s1:]
a = t.index("template <int A, int M>\n__device__ __forceinline__ void s4b_tail(")
b = t.index("\n}\n", a)
body = t[a:b]
body = body.replace("template <int A, int M>\n__device__", "template <int A, int M, bool L2R>\n__device__", 1)
old_ld = "    const bf16 *q_ = p.q_row + cb, *k_ = p.k + cb;\n"
assert body.count(old_ld) == 1
body = body.replace(old_ld, old_ld + """    float rq = 1.f, rk = 1.f;                                   // kra P-13: per-lane row rstd
    if constexpr (L2R) {
        if (rok) { const size_t ri = (size_t)(tok0 + row) * H + hh; rq = p.q_rstd[ri]; rk = p.k_rstd[ri]; }
    }
""", 1)
old_kq = """        const v4bh kk = rok ? ld8(k_ + ro + kc) : v4bh{0, 0, 0, 0};
        const v4bh qq = rok ? ld8(q_ + ro + kc) : v4bh{0, 0, 0, 0};"""
assert body.count(old_kq) == 1
body = body.replace(old_kq, """        v4bh kk = rok ? ld8(k_ + ro + kc) : v4bh{0, 0, 0, 0};
        v4bh qq = rok ? ld8(q_ + ro + kc) : v4bh{0, 0, 0, 0};
        if constexpr (L2R) { kk = l2n4(kk, rk); qq = l2n4(qq, rq); }   // kra P-13""", 1)
t = t[:a] + body + t[b:]
old_k = "template <bool VL>\n__global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(1, 3)))\np3_wy_s4b_kernel(WyS4BParams p) {"
assert t.count(old_k) == 1
t = t.replace(old_k, old_k.replace("template <bool VL>", "template <bool VL, bool L2R = false>"))
for c in range(4):
    lab = f"case {c}:" if c < 3 else "default:"
    o = f"{lab} s4b_tail<{c}, M>(p, sm, aa, db0, beta_t, tok0, cl); break;"
    assert t.count(o) == 1, o
    t = t.replace(o, f"{lab} s4b_tail<{c}, M, L2R>(p, sm, aa, db0, beta_t, tok0, cl); break;")
old_sig = """extern "C" int g2_wy_s4_split(const void* q_kgb, const void* k, const void* v_vb,
                              const void* v_new, const void* g, const void* beta,
                              const void* A, const void* h, const void* do_, const void* dh,
                              const void* dv, const void* dAqk, void* dq, void* dk,
                              void* dv2, void* dg, void* db, void* dA, void* dwT,
                              void* db_part, float scale, int T, int H, int NT,
                              const void* q_row, const void* v_row, const int* ctok,
                              const int* clen, hipStream_t s) {"""
assert t.count(old_sig) == 1
t = t.replace(old_sig, old_sig.replace("g2_wy_s4_split(", "g2_wy_s4_split2(").replace(
    "const int* clen, hipStream_t s) {", "const int* clen, const void* q_rstd, const void* k_rstd,\n                               hipStream_t s) {"))
old_lb = """        if (ctok) hipLaunchKernelGGL(p3_wy_s4b_kernel<true>, dim3(NT, H), dim3(256), 0, s, pb);
        else hipLaunchKernelGGL(p3_wy_s4b_kernel<false>, dim3(NT, H), dim3(256), 0, s, pb);
        return (int)hipGetLastError();
    }
}"""
assert t.count(old_lb) == 1
t = t.replace(old_lb, """        pb.q_rstd = (const float*)q_rstd; pb.k_rstd = (const float*)k_rstd;   // kra P-13
        if (q_rstd != nullptr) {
            if (ctok) hipLaunchKernelGGL((p3_wy_s4b_kernel<true, true>), dim3(NT, H), dim3(256), 0, s, pb);
            else hipLaunchKernelGGL((p3_wy_s4b_kernel<false, true>), dim3(NT, H), dim3(256), 0, s, pb);
        } else {
            if (ctok) hipLaunchKernelGGL(p3_wy_s4b_kernel<true>, dim3(NT, H), dim3(256), 0, s, pb);
            else hipLaunchKernelGGL(p3_wy_s4b_kernel<false>, dim3(NT, H), dim3(256), 0, s, pb);
        }
        return (int)hipGetLastError();
    }
}

""" + old_sig + """
    return g2_wy_s4_split2(q_kgb, k, v_vb, v_new, g, beta, A, h, do_, dh, dv, dAqk, dq, dk, dv2, dg,
                           db, dA, dwT, db_part, scale, T, H, NT, q_row, v_row, ctok, clen,
                           nullptr, nullptr, s);
}""")
Y.write_text(t)

# ---------------------------------------------------------------- bwd_tail
TL = K / "g2_bwd_tail.cuh"
sub(TL, "    int gate_none; int beta_f32;\n};",
    "    int gate_none; int beta_f32;\n    int y_raw;   // kra P-13: qn/kn hold RAW q/k -> y = bf_rne(bf2f(x) * rstd) at load\n};")
sub(TL, """                            r4[w2][u] = rs[(size_t)(row + u) * p.H + hh];
""", """                            r4[w2][u] = rs[(size_t)(row + u) * p.H + hh];
                            if (p.y_raw) {                          // kra P-13
                                yv[w2][u].x = bf_rne(bf2f(yv[w2][u].x) * r4[w2][u]);
                                yv[w2][u].y = bf_rne(bf2f(yv[w2][u].y) * r4[w2][u]);
                            }
""")
old_t = """                           int B, int T, int H, const int* ctok, const int* clen, int NTv,
                           int gate_none, int beta_f32, hipStream_t s) {"""
t = TL.read_text()
assert t.count(old_t) == 1
head = t[t.index('extern "C" int g2_bwd_tail('):t.index(old_t)]
t = t.replace('extern "C" int g2_bwd_tail(', 'extern "C" int g2_bwd_tail2(', 1)
t = t.replace(old_t, old_t.replace("int gate_none, int beta_f32, hipStream_t s) {",
                                   "int gate_none, int beta_f32, int y_raw, hipStream_t s) {"), 1)
old_init = "                     use_l2, use_beta, H, NT, ctok, clen, gate_none, beta_f32};"
assert t.count(old_init) == 1
t = t.replace(old_init, old_init + "\n    p.y_raw = y_raw;   // kra P-13")
i = t.index('extern "C" int g2_bwd_tail2(')
j = t.index("\n}\n", i) + 3
t = t[:j] + "\n" + head + old_t + """
    return g2_bwd_tail2(dq, dk, dg, db, qn, kn, q_rstd, k_rstd, g_raw, A_log, dt_bias, beta_raw,
                        dqo, dko, dg_raw, dA_log, ddt_bias, db_out, dA_part, ddt_part,
                        lower_bound, beta_scale, use_l2, use_beta, B, T, H, ctok, clen, NTv,
                        gate_none, beta_f32, 0, s);
}
""" + t[j:]
TL.write_text(t)

if KONLY:
    print("P-13 KERNELS APPLIED")
    sys.exit(0)

# ---------------------------------------------------------------- bindings
BD = K / "g2_bindings.cpp"
sub(BD, "extern \"C\" int g2_wu(const void* k,",
    """extern "C" int g2_wu2(const void* k, const void* v, const void* q, const void* beta,
                      const void* A, const void* g,
                      void* Wn, void* Wb, void* Un, void* KgA, void* Kn, void* Qn,
                      void* kgB, void* vB,
                      int B, int T, int H, const int* ctok, const int* clen,
                      const void* q_rstd, const void* k_rstd, hipStream_t s);
extern "C" int g2_wy_s4_split2(const void* q_kgb, const void* k, const void* v_vb,
                               const void* v_new, const void* g, const void* beta,
                               const void* A, const void* h, const void* do_, const void* dh,
                               const void* dv, const void* dAqk, void* dq, void* dk,
                               void* dv2, void* dg, void* db, void* dA, void* dwT,
                               void* db_part, float scale, int T, int H, int NT,
                               const void* q_row, const void* v_row, const int* ctok,
                               const int* clen, const void* q_rstd, const void* k_rstd,
                               hipStream_t s);
extern "C" int g2_bwd_tail2(const void* dq, const void* dk, const void* dg,
                            const void* db, const void* qn, const void* kn,
                            const void* q_rstd, const void* k_rstd,
                            const void* g_raw, const void* A_log,
                            const void* dt_bias, const void* beta_raw,
                            void* dqo, void* dko, void* dg_raw, void* dA_log,
                            void* ddt_bias, void* db_out,
                            void* dA_part, void* ddt_part,
                            double lower_bound, double beta_scale,
                            int use_l2, int use_beta,
                            int B, int T, int H, const int* ctok, const int* clen, int NTv,
                            int gate_none, int beta_f32, int y_raw, hipStream_t s);
extern "C" int g2_wu(const void* k,""")
RS = """    auto rptr = [&](const c10::optional<torch::Tensor>& r, const char* what) -> const void* {
        if (!r.has_value() || !r->defined()) return nullptr;            // kra P-13
        check_contig(*r, what); check_f32(*r, what);
        TORCH_CHECK(r->numel() * 128 == k.numel(), "g2: rstd must hold one fp32 per q/k row");
        return r->data_ptr();
    };
"""
sub(BD, """                              c10::optional<torch::Tensor> clen, bool emit_p1) {
    TORCH_CHECK(k.dim() == 4 && k.size(0) == 1, "g2_wu: B=1 only (batch = varlen rows)");""",
    """                              c10::optional<torch::Tensor> clen, bool emit_p1,
                              c10::optional<torch::Tensor> q_rstd,
                              c10::optional<torch::Tensor> k_rstd) {
    TORCH_CHECK(k.dim() == 4 && k.size(0) == 1, "g2_wu: B=1 only (batch = varlen rows)");
""" + RS)
sub(BD, """    const int rc = g2_wu(k.data_ptr(), v.data_ptr(), q.data_ptr(), beta.data_ptr(),
                         A.data_ptr(), g.data_ptr(), emit_p1 ? Wn.data_ptr() : nullptr, Wb.data_ptr(),
                         emit_p1 ? Un.data_ptr() : nullptr, KgA.data_ptr(),
                         emit_p1 ? Kn.data_ptr() : nullptr, Qn.data_ptr(),
                         kgB.data_ptr(), vB.data_ptr(),
                         1, (int)(chunks * 64), (int)heads, rm.ctok, rm.clen, cur_stream());""",
    """    const void* qr = rptr(q_rstd, "wu q_rstd");
    const void* kr = rptr(k_rstd, "wu k_rstd");
    TORCH_CHECK((qr == nullptr) == (kr == nullptr), "g2_wu: pass both rstd or neither");
    const int rc = g2_wu2(k.data_ptr(), v.data_ptr(), q.data_ptr(), beta.data_ptr(),
                          A.data_ptr(), g.data_ptr(), emit_p1 ? Wn.data_ptr() : nullptr, Wb.data_ptr(),
                          emit_p1 ? Un.data_ptr() : nullptr, KgA.data_ptr(),
                          emit_p1 ? Kn.data_ptr() : nullptr, Qn.data_ptr(),
                          kgB.data_ptr(), vB.data_ptr(),
                          1, (int)(chunks * 64), (int)heads, rm.ctok, rm.clen, qr, kr, cur_stream());""")
sub(BD, """          py::arg("clen") = py::none(), py::arg("emit_p1") = true);""",
    """          py::arg("clen") = py::none(), py::arg("emit_p1") = true,
          py::arg("q_rstd") = py::none(), py::arg("k_rstd") = py::none());""")
# wy_s4_split
sub(BD, """                                       c10::optional<torch::Tensor> ctok,
                                       c10::optional<torch::Tensor> clen) {
    for (auto* t : {&q_kgb, &k, &v_vb, &v_new, &A, &h, &do_, &dh, &dv, &q_row, &v_row}) {""",
    """                                       c10::optional<torch::Tensor> ctok,
                                       c10::optional<torch::Tensor> clen,
                                       c10::optional<torch::Tensor> q_rstd,
                                       c10::optional<torch::Tensor> k_rstd) {
""" + RS + """    for (auto* t : {&q_kgb, &k, &v_vb, &v_new, &A, &h, &do_, &dh, &dv, &q_row, &v_row}) {""")
sub(BD, """    const int rc = g2_wy_s4_split(q_kgb.data_ptr(), k.data_ptr(), v_vb.data_ptr(), v_new.data_ptr(),""",
    """    const void* qr = rptr(q_rstd, "wy_s4_split q_rstd");
    const void* kr = rptr(k_rstd, "wy_s4_split k_rstd");
    TORCH_CHECK((qr == nullptr) == (kr == nullptr), "wy_s4_split: pass both rstd or neither");
    const int rc = g2_wy_s4_split2(q_kgb.data_ptr(), k.data_ptr(), v_vb.data_ptr(), v_new.data_ptr(),""")
sub(BD, """                                  q_row.data_ptr(), v_row.data_ptr(), rm.ctok, rm.clen,
                                  cur_stream());
    TORCH_CHECK(rc == 0, "g2_wy_s4_split launch failed: ", rc);""",
    """                                  q_row.data_ptr(), v_row.data_ptr(), rm.ctok, rm.clen,
                                  qr, kr, cur_stream());
    TORCH_CHECK(rc == 0, "g2_wy_s4_split launch failed: ", rc);""")
# bwd_tail
sub(BD, """                                    c10::optional<torch::Tensor> clen, bool gate_none) {""",
    """                                    c10::optional<torch::Tensor> clen, bool gate_none,
                                    bool y_raw) {""")
sub(BD, """    const int rc = g2_bwd_tail(dq.data_ptr(), dk.data_ptr(), dg.data_ptr(),""",
    """    TORCH_CHECK(!y_raw || use_l2, "bwd_tail: y_raw needs use_l2 (rstd)");   // kra P-13
    const int rc = g2_bwd_tail2(dq.data_ptr(), dk.data_ptr(), dg.data_ptr(),""")
sub(BD, """                               gate_none ? 1 : 0, beta_f32 ? 1 : 0, cur_stream());
    TORCH_CHECK(rc == 0, "g2_bwd_tail failed: ", rc);""",
    """                               gate_none ? 1 : 0, beta_f32 ? 1 : 0, y_raw ? 1 : 0, cur_stream());
    TORCH_CHECK(rc == 0, "g2_bwd_tail failed: ", rc);""")

# ---------------------------------------------------------------- python wrappers
OPS = ROOT / "hip_kda" / "ops" / "g2_ops.py"
sub(OPS, """def wu(k, v, q, beta, A, g, heads, chunks, rowmap=None, emit_p1=True):""",
    """def wu(k, v, q, beta, A, g, heads, chunks, rowmap=None, emit_p1=True, q_rstd=None, k_rstd=None):""")
sub(OPS, """                     *_rowmap(rowmap)[:2], bool(emit_p1))""",
    """                     *_rowmap(rowmap)[:2], bool(emit_p1),
                     q_rstd=None if q_rstd is None else q_rstd.contiguous(),   # kra P-13: raw q/k
                     k_rstd=None if k_rstd is None else k_rstd.contiguous())""")
sub(OPS, """def bwd_tail(dq, dk, dg, db, qn, kn, q_rstd, k_rstd, g_raw, a_log, dt_bias,
             beta_raw, lower_bound, beta_scale, use_l2, use_beta, rowmap=None, gate_none=False):""",
    """def bwd_tail(dq, dk, dg, db, qn, kn, q_rstd, k_rstd, g_raw, a_log, dt_bias,
             beta_raw, lower_bound, beta_scale, use_l2, use_beta, rowmap=None, gate_none=False,
             y_raw=False):""")
sub(OPS, """                           bool(use_l2), bool(use_beta), *_rowmap(rowmap)[:2], bool(gate_none))""",
    """                           bool(use_l2), bool(use_beta), *_rowmap(rowmap)[:2], bool(gate_none),
                           bool(y_raw))   # kra P-13: qn/kn are raw q/k""")
sub(OPS, """                q_row, v_row, dAqk, heads, tokens, scale, emit_dA=False, rowmap=None):""",
    """                q_row, v_row, dAqk, heads, tokens, scale, emit_dA=False, rowmap=None,
                q_rstd=None, k_rstd=None):""")
sub(OPS, """                              float(scale), bool(emit_dA), *_rowmap(rowmap)[:2])""",
    """                              float(scale), bool(emit_dA), *_rowmap(rowmap)[:2],
                              None if q_rstd is None else q_rstd.contiguous(),   # kra P-13
                              None if k_rstd is None else k_rstd.contiguous())""")

# ---------------------------------------------------------------- k3_g2 wiring
KG = ROOT / "hip_kda" / "ops" / "k3_g2.py"
sub(KG, """        qn, kn = q, k
        if ctx.use_qk_l2norm_in_kernel:
            if os.environ.get("KDA_G2_BL2", "1").strip() != "0":
                # V1: the forward already saved q_rstd/k_rstd; bf16(x*rstd) is
                # bitwise equal to l2norm_fwd (unit-gated at 3 shapes) -> one
                # HIP kernel replaces two Triton l2norm_fwd calls.
                qn, kn = g2_ops.l2n_apply(q, k, q_rstd, k_rstd)""",
    """        qn, kn = q, k
        # kra P-13: on the production route (S4 split + HIP / T5 tail) the three consumers of
        # qn/kn (wu, wy_s4b, bwd_tail) recompute bf16(x*rstd) at load from the raw q/k --
        # the l2n_apply launch (its write + every re-read) disappears.  KDA_G2_L2R=0 restores it.
        l2r = False
        if ctx.use_qk_l2norm_in_kernel:
            if os.environ.get("KDA_G2_BL2", "1").strip() != "0":
                t5_pred = (os.environ.get("KDA_G2_T5", "1").strip() != "0"
                           and not ctx.precomputed_g
                           and g_raw.dtype == torch.bfloat16
                           and a_log.dtype == torch.float32
                           and dt_bias.dtype == torch.float32
                           and (not ctx.use_beta_sigmoid_in_kernel
                                or beta_raw.dtype == torch.bfloat16))
                l2r = (os.environ.get("KDA_G2_L2R", "1").strip() != "0"
                       and os.environ.get("KDA_G2_S4", "1").strip() != "0"
                       and os.environ.get("KDA_G2_P4", "3").strip() == "3"
                       and (hipi or t5_pred)
                       and q_rstd is not None and k_rstd is not None)
                # V1: the forward already saved q_rstd/k_rstd; bf16(x*rstd) is
                # bitwise equal to l2norm_fwd (unit-gated at 3 shapes) -> one
                # HIP kernel replaces two Triton l2norm_fwd calls.
                if not l2r:
                    qn, kn = g2_ops.l2n_apply(q, k, q_rstd, k_rstd)""")
sub(KG, """        Wn, Wb, Un, KgA, Kn, Qn, kgB, vB = g2_ops.wu(kn, v, qn, beta, Akk, g, H, NT, rowmap=lay,
                                                     emit_p1=not kept)""",
    """        l2r_kw = dict(q_rstd=q_rstd, k_rstd=k_rstd) if l2r else {}   # kra P-13
        Wn, Wb, Un, KgA, Kn, Qn, kgB, vB = g2_ops.wu(kn, v, qn, beta, Akk, g, H, NT, rowmap=lay,
                                                     emit_p1=not kept, **l2r_kw)""")
sub(KG, """                dq, dk, dv_grad, dg, db = g2_ops.wy_s4_split(*wy_args, dAqk, H, 64 * NT, scale,
                                                             rowmap=lay)""",
    """                dq, dk, dv_grad, dg, db = g2_ops.wy_s4_split(*wy_args, dAqk, H, 64 * NT, scale,
                                                             rowmap=lay, **l2r_kw)""")
sub(KG, """                ctx.use_qk_l2norm_in_kernel, ctx.use_beta_sigmoid_in_kernel, rowmap=lay,
                gate_none=pg)""",
    """                ctx.use_qk_l2norm_in_kernel, ctx.use_beta_sigmoid_in_kernel, rowmap=lay,
                gate_none=pg, y_raw=l2r)""")
sub(KG, """                beta_raw, lower_bound, 2.0 if ctx.allow_neg_eigval else 1.0,
                ctx.use_qk_l2norm_in_kernel, ctx.use_beta_sigmoid_in_kernel, rowmap=lay)
        else:""",
    """                beta_raw, lower_bound, 2.0 if ctx.allow_neg_eigval else 1.0,
                ctx.use_qk_l2norm_in_kernel, ctx.use_beta_sigmoid_in_kernel, rowmap=lay,
                y_raw=l2r)
        else:""")
print("P-13 APPLIED")
