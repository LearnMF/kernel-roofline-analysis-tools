"""kra P-52 (A-class): Gn carries the chunk decay factors exp2f(G_last) instead of G_last.

Every consumer of Gn (the forward / recompute recurrences and dhu, 7 sites incl. legacy engines)
uses it only as d = exp2f(gl) -- 8 range-checked exp2f per step per lane in the forward
recurrence (~50 of its 433 loop instructions: v_cmp/v_cndmask/v_exp/v_mul/v_add), on the serial
chain.  The producers now apply the SAME device exp2f once per element:
  * forward: fwd_prep (prep_a) stores exp2f(refv) at its Gn site
  * backward / FI=3 forward: gn.to_gn and k3_g2._to_gn_varlen run a device exp2f pass (g2_exp2_)
Same function on the same fp32 input -> bit-identical decays, no extra memory (Gn is replaced).
The consumers take the decay through g2_vcopy (an asm-volatile v_mov) so that the production
placement of the Gn prefetch (mid-step) and its VMEM pacing are kept.
The legacy layout-kernel path convert(Mode2.ToGn) keeps raw G (not on any production route).
    python3 p52_gn_exp2.py <tree>"""
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
G2 = ROOT / "csrc" / "g2"


def sub(path, old, new, cnt):
    t = path.read_text()
    assert t.count(old) == cnt, (path.name, old[:80], t.count(old))
    path.write_text(t.replace(old, new))


# consumers: d = exp2f(gl) -> d = gl
sub(G2 / "g2_fwd_v8.cuh", "for (int e = 0; e < 4; ++e) d[e] = exp2f(gl[e]);",
    "for (int e = 0; e < 4; ++e) d[e] = g2_vcopy(gl[e]);   // kra P-52: Gn = exp2f(G)", 1)
sub(G2 / "g2_fwd_v8.cuh", "for (int e = 0; e < 4; ++e) { d0[e] = exp2f(gl0[e]); d1[e] = exp2f(gl1[e]); }",
    "for (int e = 0; e < 4; ++e) { d0[e] = g2_vcopy(gl0[e]); d1[e] = g2_vcopy(gl1[e]); }   // kra P-52: Gn = exp2f(G)", 2)
sub(G2 / "g2_engine.cuh", "for (int e = 0; e < 4; ++e) d[e] = exp2f(gl[kl][e]);",
    "for (int e = 0; e < 4; ++e) d[e] = g2_vcopy(gl[kl][e]);   // kra P-52: Gn = exp2f(G)", 1)
sub(G2 / "g2_engine.cuh", "for (int e = 0; e < 4; ++e) d[e] = exp2f(gl[e]);",
    "for (int e = 0; e < 4; ++e) d[e] = g2_vcopy(gl[e]);   // kra P-52: Gn = exp2f(G)", 2)
sub(G2 / "g2_engine.cuh", "for (int e = 0; e < 4; ++e) { d0[e] = exp2f(gl0[e]); d1[e] = exp2f(gl1[e]); }",
    "for (int e = 0; e < 4; ++e) { d0[e] = g2_vcopy(gl0[e]); d1[e] = g2_vcopy(gl1[e]); }   // kra P-52: Gn = exp2f(G)", 1)
for f in ("g2_fwd_v8.cuh", "g2_engine.cuh"):
    t = (G2 / f).read_text()
    assert not re.search(r"exp2f\(gl", t), f
# opaque copy: with a plain d = gl the compiler folds the decay into the step-end FMA, keeps gl
# live across the mid-step loadG(nxt) and pays a loop-carried register copy that waits on the
# prefetch every step (MODE 4: 0.84x at 16K/48).  An asm-volatile v_mov at the former exp2f site
# ends gl's live range there, so the production load placement / VMEM pacing stays unchanged.
sub(G2 / "g2_common.h", "__device__ __forceinline__ int pi16(int n) { return 4 * (n & 3) + (n >> 2); }",
    """__device__ __forceinline__ int pi16(int n) { return 4 * (n & 3) + (n >> 2); }
// kra P-52: opaque register copy (ends the source's live range at this point; value unchanged)
__device__ __forceinline__ float g2_vcopy(float x) {
    float y;
    asm volatile("v_mov_b32 %0, %1" : "=v"(y) : "v"(x));
    return y;
}""", 1)
# forward producer
sub(G2 / "g2_fwd_prep.cuh", "gnp[(col >> 4) * 16 + (col & 3) * 4 + ((col & 15) >> 2)] = refv[e];",
    "gnp[(col >> 4) * 16 + (col & 3) * 4 + ((col & 15) >> 2)] = exp2f(refv[e]);   // kra P-52", 1)
# device exp2f pass (backward producer)
sub(G2 / "g2_fwd_v8.cuh", 'extern "C" int g2_fwd_h_v8o(const void* Wn, const void* Kn, const void* Un, const void* Gn,',
    '''// kra P-52: Gn -> exp2f(Gn) in place, the consumers' former per-step exp2f evaluated once
__global__ void __launch_bounds__(256) g2_exp2_kernel(float* __restrict__ x, long n) {
    const long i = (long)blockIdx.x * 256 + threadIdx.x;
    if (i < n) x[i] = exp2f(x[i]);
}
extern "C" int g2_exp2_inplace(float* x, long n, hipStream_t s) {
    if (n <= 0) return 0;
    hipLaunchKernelGGL(g2_exp2_kernel, dim3((unsigned)((n + 255) / 256)), dim3(256), 0, s, x, n);
    return (int)hipGetLastError();
}

extern "C" int g2_fwd_h_v8o(const void* Wn, const void* Kn, const void* Un, const void* Gn,''', 1)
B = G2 / "g2_bindings.cpp"
sub(B, 'extern "C" int g2_dhu_v8b(const void* KgA, const void* Qg, const void* Wb, const void* dOT, const void* dv,',
    'extern "C" int g2_exp2_inplace(float* x, long n, hipStream_t s);   // kra P-52\n'
    'extern "C" int g2_dhu_v8b(const void* KgA, const void* Qg, const void* Wb, const void* dOT, const void* dv,', 1)
sub(B, "std::vector<torch::Tensor> dhu_bn(torch::Tensor KgA, torch::Tensor Qn, torch::Tensor Wb,",
    """// kra P-52: in-place device exp2f (Gn -> chunk decay factors)
torch::Tensor exp2_(torch::Tensor x) {
    check_contig(x, "exp2_"); check_f32(x, "exp2_");
    c10::hip::HIPGuard guard(x.get_device());
    const int rc = g2_exp2_inplace(x.data_ptr<float>(), (long)x.numel(), cur_stream());
    TORCH_CHECK(rc == 0, "g2_exp2_inplace failed: ", rc);
    return x;
}

std::vector<torch::Tensor> dhu_bn(torch::Tensor KgA, torch::Tensor Qn, torch::Tensor Wb,""", 1)
sub(B, '    m.def("fwd_h_bn", &fwd_h_bn, "G2 P1 recompute emitting bN(h)",',
    '    m.def("exp2_", &exp2_, "G2: in-place device exp2f (Gn -> decay factors) [kra P-52]");\n'
    '    m.def("fwd_h_bn", &fwd_h_bn, "G2 P1 recompute emitting bN(h)",', 1)
# python producers
NAT = ROOT / "hip_kda" / "ops" / "g2_native.py"
sub(NAT, '''def to_gn(g: torch.Tensor, H: int) -> torch.Tensor:
    """Per-chunk last g row, fp32: [B,T,H,K] -> (B,H,NT,8,4,4)."""
    B, T = g.shape[0], g.shape[1]
    NT = T // BT
    gl = g[:, BT - 1::BT].contiguous()
    return _perm(gl, (B, NT, H, 8, 4, 4), (0, 2, 1, 3, 5, 4))''',
    '''def to_gn(g: torch.Tensor, H: int) -> torch.Tensor:
    """Per-chunk decay factors exp2f(last g row), fp32: [B,T,H,K] -> (B,H,NT,8,4,4).
    kra P-52: the engines read the decay directly; the device exp2f here is the one they ran."""
    B, T = g.shape[0], g.shape[1]
    NT = T // BT
    gl = g[:, BT - 1::BT].contiguous()
    from . import g2_ops as _ops
    return _ops._ext().exp2_(_perm(gl, (B, NT, H, 8, 4, 4), (0, 2, 1, 3, 5, 4)))''', 1)
K = ROOT / "hip_kda" / "ops" / "k3_g2.py"
sub(K, '''    gl = g.index_select(1, lay.last_tok).contiguous()      # [1, NT, H, 128]
    return gn._perm(gl, (1, lay.NT, H, 8, 4, 4), (0, 2, 1, 3, 5, 4))''',
    '''    gl = g.index_select(1, lay.last_tok).contiguous()      # [1, NT, H, 128]
    # kra P-52: decay factors exp2f(G) (the engines' former per-step exp2f, evaluated once)
    return g2_ops._ext().exp2_(gn._perm(gl, (1, lay.NT, H, 8, 4, 4), (0, 2, 1, 3, 5, 4)))''', 1)

# ---- deep-saturation exception: the narrow MODE-4 recompute at >= 192 CTAs measured 0.93x (16K/48)
# and 0.83x (8K/96) without the exp2f VALU between its loads (R21: denser VMEM issue at high
# occupancy).  There it keeps the production instance: GE=false reads the raw Gn (GnRaw, a 512 B
# per chunk*head copy the backward keeps) and runs exp2f in-kernel -- the production code verbatim.
F8 = G2 / "g2_fwd_v8.cuh"
sub(F8, "    const int* ctok; const int* clen;\n};",
    "    const int* ctok; const int* clen;\n    const float* GnRaw;                                // kra P-52: raw G_last (GE=false instance)\n};", 1)
sub(F8, "template <bool VL, int MODE = 0>\n__global__ void __launch_bounds__(256) g2_fwd_h_v8o4_kernel(FwdOParams p) {",
    "template <bool VL, int MODE = 0, bool GE = true>   // kra P-52: GE = Gn holds exp2f(G)\n__global__ void __launch_bounds__(256) g2_fwd_h_v8o4_kernel(FwdOParams p) {", 1)
_t = F8.read_text()
_k0 = _t.index("g2_fwd_h_v8o4_kernel(FwdOParams p) {")
_k1 = _t.index("g2_fwd_h_v8o4w_kernel(FwdOParams p) {")
_body = _t[_k0:_k1]
_old = "#pragma unroll\n            for (int e = 0; e < 4; ++e) { d0[e] = g2_vcopy(gl0[e]); d1[e] = g2_vcopy(gl1[e]); }   // kra P-52: Gn = exp2f(G)"
assert _body.count(_old) == 1
_new = ("if constexpr (GE) {   // kra P-52: decays precomputed\n"
        "                #pragma unroll\n"
        "                for (int e = 0; e < 4; ++e) { d0[e] = g2_vcopy(gl0[e]); d1[e] = g2_vcopy(gl1[e]); }\n"
        "            } else {               // production instance (raw G_last)\n"
        "                #pragma unroll\n"
        "                for (int e = 0; e < 4; ++e) { d0[e] = exp2f(gl0[e]); d1[e] = exp2f(gl1[e]); }\n"
        "            }")
F8.write_text(_t[:_k0] + _body.replace(_old, _new) + _t[_k1:])
sub(F8, "                            const int* ctok, const int* clen, hipStream_t s) {\n    // T is the chunk-padded row count",
    "                            const int* ctok, const int* clen, const void* GnRaw, hipStream_t s) {\n    // T is the chunk-padded row count", 1)
sub(F8, "                 scale, H, T / kBT, seg, nseg, ctok, clen};",
    "                 scale, H, T / kBT, seg, nseg, ctok, clen, (const float*)GnRaw};", 1)
sub(F8, "        hipLaunchKernelGGL(kern, grid, dim3(256), dyn, s, p);\n    };\n",
    "        hipLaunchKernelGGL(kern, grid, dim3(256), dyn, s, p);\n    };\n"
    "    // kra P-52: narrow MODE 4 at deep saturation keeps the production instance on the raw Gn\n"
    "    const bool raw4 = GnRaw != nullptr && 4L * nseg * H >= g2_env_long(\"KDA_G2_BN_RAW_MIN_CTAS\", 192);\n"
    "    auto launch4raw = [&](auto kern) {\n"
    "        FwdOParams q = p; q.Gn = (const float*)GnRaw;\n"
    "        const size_t dyn = throttle ? g2_lds_topup(kern, 24 * 1024) : 0;\n"
    "        hipLaunchKernelGGL(kern, grid, dim3(256), dyn, s, q);\n"
    "    };\n", 1)
sub(F8, "            case 4: launch4(g2_fwd_h_v8o4_kernel<VL, 4>); break;",
    "            case 4: if (raw4) launch4raw(g2_fwd_h_v8o4_kernel<VL, 4, false>);\n"
    "                    else      launch4(g2_fwd_h_v8o4_kernel<VL, 4>);\n"
    "                    break;", 1)
B2 = G2 / "g2_bindings.cpp"
sub(B2, "float scale, int B, int T, int H, const int* seg, int nseg,\n                            const int* ctok, const int* clen, hipStream_t s);",
    "float scale, int B, int T, int H, const int* seg, int nseg,\n                            const int* ctok, const int* clen, const void* GnRaw, hipStream_t s);", 1)
sub(B2, "                                (int)B, (int)T, (int)H, sg.first, sg.second, rm.ctok, rm.clen,\n                                cur_stream());",
    "                                (int)B, (int)T, (int)H, sg.first, sg.second, rm.ctok, rm.clen,\n                                nullptr, cur_stream());", 1)
sub(B2, "                                    torch::Tensor Gn, c10::optional<torch::Tensor> seg,\n                                    bool emit_an) {",
    "                                    torch::Tensor Gn, c10::optional<torch::Tensor> seg,\n                                    bool emit_an, c10::optional<torch::Tensor> gn_raw) {", 1)
sub(B2, "                                (int)H, sgb.first, sgb.second, nullptr, nullptr, cur_stream());",
    "                                (int)H, sgb.first, sgb.second, nullptr, nullptr,\n"
    "                                gn_raw.has_value() ? gn_raw->data_ptr() : nullptr, cur_stream());", 1)
sub(B2, "         py::arg(\"seg\") = py::none(), py::arg(\"emit_an\") = true);",
    "         py::arg(\"seg\") = py::none(), py::arg(\"emit_an\") = true, py::arg(\"gn_raw\") = py::none());", 1)
OPS = ROOT / "hip_kda" / "ops" / "g2_ops.py"
sub(OPS, "def fwd_h_bn(wn, kn, un, gn_t, seg=None, emit_an=True):",
    "def fwd_h_bn(wn, kn, un, gn_t, seg=None, emit_an=True, gn_raw=None):", 1)
sub(OPS, "    return _ext().fwd_h_bn(wn, kn, un, gn_t, seg, bool(emit_an))  # -> (HbN, Vn, ANv)",
    "    # kra P-52: gn_raw (raw G_last) lets the deep-saturation narrow instance keep the production code\n"
    "    return _ext().fwd_h_bn(wn, kn, un, gn_t, seg, bool(emit_an), gn_raw)  # -> (HbN, Vn, ANv)", 1)
sub(NAT, "def to_gn(g: torch.Tensor, H: int) -> torch.Tensor:",
    "def to_gn(g: torch.Tensor, H: int, raw: bool = False) -> torch.Tensor:", 1)
sub(NAT, "    from . import g2_ops as _ops\n    return _ops._ext().exp2_(_perm(gl, (B, NT, H, 8, 4, 4), (0, 2, 1, 3, 5, 4)))",
    "    t = _perm(gl, (B, NT, H, 8, 4, 4), (0, 2, 1, 3, 5, 4))\n"
    "    if raw:                                   # raw G_last (the GE=false recompute instance)\n"
    "        return t\n"
    "    from . import g2_ops as _ops\n"
    "    return _ops._ext().exp2_(t)", 1)
sub(K, "def _to_gn_varlen(g, H, lay):", "def _to_gn_varlen(g, H, lay, raw=False):", 1)
sub(K, "    return g2_ops._ext().exp2_(gn._perm(gl, (1, lay.NT, H, 8, 4, 4), (0, 2, 1, 3, 5, 4)))",
    "    t = gn._perm(gl, (1, lay.NT, H, 8, 4, 4), (0, 2, 1, 3, 5, 4))\n"
    "    return t if raw else g2_ops._ext().exp2_(t)", 1)
sub(K, "        Gn = gn.to_gn(g, H) if lay is None else _to_gn_varlen(g, H, lay)\n\n        # kra P-22",
    "        # kra P-52: Gn = exp2f(G_last) decays; Gn_raw feeds the deep-saturation recompute instance\n"
    "        Gn_raw = gn.to_gn(g, H, raw=True) if lay is None else _to_gn_varlen(g, H, lay, raw=True)\n"
    "        Gn = g2_ops._ext().exp2_(Gn_raw.clone())\n\n        # kra P-22", 1)
sub(K, "            bN_h, Vn, aN_vn = g2_ops.fwd_h_bn(Wn, Kn, Un, Gn, seg=ctx.seg,\n"
       "                                              emit_an=not no_an)  # B2: P1 v8 (bN + aN direct)",
    "            bN_h, Vn, aN_vn = g2_ops.fwd_h_bn(Wn, Kn, Un, Gn, seg=ctx.seg,\n"
    "                                              emit_an=not no_an, gn_raw=Gn_raw)  # B2: P1 v8 (bN + aN direct)", 1)
print("P-52 APPLIED")
