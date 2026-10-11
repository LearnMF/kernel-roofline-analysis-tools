"""kra P-23 (round 7, structural, BW-L2-specific): the h recurrence (v8o4) with a 64-column V slab
per CTA (4 V-tiles; v8o4 has 2), dispatched only at deep saturation.  Every V slab of a head
reloads the head's K-side operands (W, K, and for the forward QgA/AqkA: ~56 KB per chunk step);
with 4 slabs per head the reload fits the 8 MB L2 up to ~48 heads (8K: 1.00x interface bytes,
16.6-17.7 us/head) but not beyond (8K/96: 1.47-1.85x interface bytes, ~22 us/head).  Two slabs
per head halve that traffic (and the matching VMEM instructions).  A800's 40 MB L2 absorbs the
same redundancy -- the BW-specific reason (DCU textbook 6.5.4: on gfx936 L2 locality first).
Every output tile keeps its MMAC operands and accumulation order -> bit-identical.  Production
modes only (1: forward o, 4: backward recompute without aN); KDA_G2_V8W_MIN_CTAS (default 256 =
4*nseg*H) sets the threshold, 0 forces it everywhere, a huge value disables it.
Usage: python3 p23_v8w_wide_slab.py <tree root>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_fwd_v8.cuh"
t = P.read_text()

KERNEL = r'''
// kra P-23: v8o4 with a 64-column V slab per CTA (NVT = 4 V-tiles).  At deep saturation the
// 4-slab-per-head reload of W/K/QgA/AqkA no longer fits L2; 2 slabs per head halve it.  Every
// output tile's MMAC operands and accumulation order are those of v8o4 -> bit-identical.
template <bool VL, int MODE>
__global__ void __launch_bounds__(256) g2_fwd_h_v8w_kernel(FwdOParams p) {
    static_assert(MODE == 1 || MODE == 4, "v8w: production modes only");
    constexpr int NVT = 4, kBVW = 16 * NVT;
    const bool hasOb = MODE == 1;
    const bool hasBN = MODE == 4;
    const bool hasVn = MODE == 4;
    __shared__ __align__(16) bf16 hB[kBVW * kPadK];
    __shared__ __align__(16) bf16 vB[kBVW * kPadT];
    __shared__ __align__(16) bf16 oStage[4][256];
    const int tid = threadIdx.x, wv = tid >> 6, l = tid & 63;   // wv in 0..3
    const int vs = blockIdx.x;                                  // 64-column slab
    const int sg = blockIdx.y / p.H, bh = blockIdx.y - sg * p.H;
    const int c0 = p.seg ? p.seg[sg] : 0, c1 = p.seg ? p.seg[sg + 1] : p.NT;
    if (c0 >= c1) return;
    const size_t chunk = (size_t)bh * p.NT;
    const int r1 = wv;                       // GEMM1/o T-tile
    const int kb0 = wv, kb1 = wv + 4;        // GEMM2 K-blocks
    const int m = l & 15, q4 = l >> 4;
    const size_t rowT = (size_t)p.H * kK;

    f4 ht0[NVT], ht1[NVT];
    #pragma unroll
    for (int v = 0; v < NVT; ++v) { ht0[v] = f4{0.f,0.f,0.f,0.f}; ht1[v] = ht0[v]; }
    v4bh wf[8], kf0[4], kf1[4], uf[NVT], qgf[8], aqf[4];
    f4 gl0, gl1;

    auto loadW = [&](int i) {
        const bf16* src = p.Wn + (((chunk + i) * 4 + r1) * 8) * 256 + l * 8;      // kra L16: paired
        #pragma unroll
        for (int j = 0; j < 4; ++j) ld16x2(src + j * 512, wf[2 * j], wf[2 * j + 1]);
    };
    auto loadQg = [&](int i) {
        const bf16* src = p.QgA + (((chunk + i) * 4 + r1) * 8) * 256 + l * 8;      // kra L16: paired
        #pragma unroll
        for (int j = 0; j < 4; ++j) ld16x2(src + j * 512, qgf[2 * j], qgf[2 * j + 1]);
    };
    auto loadAq = [&](int i) {
        const bf16* src = p.AqkA + (((chunk + i) * 4 + r1) * 4) * 256 + l * 8;   // kra L16: paired
        #pragma unroll
        for (int j = 0; j < 2; ++j) ld16x2(src + j * 512, aqf[2 * j], aqf[2 * j + 1]);
    };
    auto loadU = [&](int i) {          // the slab's 4 V sub-blocks = 2 L16 pairs
        #pragma unroll
        for (int j = 0; j < NVT / 2; ++j)
            ld16x2(p.Un + (((chunk + i) * 4 + r1) * 8 + NVT * vs + 2 * j) * 256 + l * 8, uf[2 * j], uf[2 * j + 1]);
    };
    auto loadG = [&](int i, int kb, f4& gl) {
        gl = *reinterpret_cast<const f4*>(p.Gn + ((chunk + i) * 8 + kb) * 16 + q4 * 4);
    };
    auto loadK = [&](int i, int kb, v4bh* kf) {
        #pragma unroll
        for (int j = 0; j < 2; ++j)                      // kra L16: paired
            ld16x2(p.Kn + (((chunk + i) * 8 + kb) * 4) * 256 + j * 512 + l * 8, kf[2 * j], kf[2 * j + 1]);
    };
    v4bh hx[NVT];
    auto write_h = [&](int kb, f4* ht) {
        #pragma unroll
        for (int vt = 0; vt < NVT; ++vt)
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                hx[vt][e] = bf_rne(ht[vt][e]);
                reinterpret_cast<short*>(hB)[(16 * vt + m) * kPadK + 16 * kb + c_col(l, e)] = hx[vt][e];
            }
    };
    auto bN_emit = [&](int i2, int kb) {
        if (hasBN) {
            #pragma unroll
            for (int vt = 0; vt < NVT; ++vt) {
                v4bh val;
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    val[e] = reinterpret_cast<const short*>(hB)[(16 * vt + 4 * q4 + e) * kPadK + 16 * kb + pi16(m)];
                const size_t off = ((((size_t)i2 * p.H + bh) * 8 + kb) * 8 + (NVT * vs + vt)) * 256 + l * 4;
                *reinterpret_cast<v4bh*>(p.HbN + off) = val;
            }
        }
    };

    loadW(c0); loadU(c0);
    loadG(c0, kb0, gl0); loadG(c0, kb1, gl1);
    loadK(c0, kb0, kf0); loadK(c0, kb1, kf1);
    write_h(kb0, ht0); write_h(kb1, ht1);
    for (int i = c0; i < c1; ++i) {
        const long otok = g2_row_tok<VL>(p.ctok, i);
        const int olen = g2_row_len<VL>(p.clen, i);
        const int nxt = i + 1 < c1 ? i + 1 : i;
        lds_barrier();                                   // hB (h_i) ready
        if (hasOb) {
            if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);   // kra P-11: keep load spacing
            loadQg(i); loadAq(i);
            if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);
        }
        bN_emit(i, kb0);
        bN_emit(i, kb1);
        // ---- GEMM1 for the slab's 4 V-tiles with ONE W load
        f4 acc[NVT];
        #pragma unroll
        for (int v = 0; v < NVT; ++v) acc[v] = f4{0.f,0.f,0.f,0.f};
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt)
            #pragma unroll
            for (int v = 0; v < NVT; ++v)
                acc[v] = mmac(wf[kt], ld8(hB + (16 * v + m) * kPadK + 16 * kt + q4 * 4), acc[v]);
        if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);
        loadW(nxt);
        if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);
        {
            v4bh x[NVT];
            #pragma unroll
            for (int e = 0; e < 4; ++e)
                #pragma unroll
                for (int v = 0; v < NVT; ++v) {
                    const float u = __uint_as_float(((uint32_t)(unsigned short)uf[v][e]) << 16);
                    x[v][e] = bf_rne(u - acc[v][e]);
                    reinterpret_cast<short*>(vB)[(16 * v + c_col(l, e)) * kPadT + 16 * r1 + m] = x[v][e];
                }
            if (hasVn) {
                #pragma unroll
                for (int v = 0; v < NVT; ++v)
                    *reinterpret_cast<v4bh*>(p.Vn + (((chunk + i) * 4 + r1) * 8 + NVT * vs + v) * 256 + l * 4) = x[v];
            }
        }
        if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);
        loadU(nxt);
        if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);
        {
            f4 d0, d1;
            #pragma unroll
            for (int e = 0; e < 4; ++e) { d0[e] = exp2f(gl0[e]); d1[e] = exp2f(gl1[e]); }
            #pragma unroll
            for (int v = 0; v < NVT; ++v) { ht0[v] *= d0; ht1[v] *= d1; }
        }
        if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);
        loadG(nxt, kb0, gl0); loadG(nxt, kb1, gl1);
        if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);
        // ---- oh for the 4 tiles (QgA/AqkA loaded ONCE)
        f4 oh[NVT];
        #pragma unroll
        for (int v = 0; v < NVT; ++v) oh[v] = f4{0.f,0.f,0.f,0.f};
        if (hasOb) {
            #pragma unroll
            for (int kt = 0; kt < 8; ++kt)
                #pragma unroll
                for (int v = 0; v < NVT; ++v)
                    oh[v] = mmac(qgf[kt], ld8(hB + (16 * v + m) * kPadK + 16 * kt + q4 * 4), oh[v]);
        }
        lds_barrier();                                   // vB (v_new_i) ready
        if (hasOb) {
            #pragma unroll
            for (int v1a = 0; v1a < NVT; ++v1a) {
                f4 oa = f4{0.f,0.f,0.f,0.f};
                #pragma unroll
                for (int st = 0; st < 4; ++st)
                    oa = mmac(aqf[st], ld8(vB + (16 * v1a + m) * kPadT + 16 * st + q4 * 4), oa);
                f4 o;
                #pragma unroll
                for (int e = 0; e < 4; ++e) o[e] = oh[v1a][e] * p.scale + oa[e];
                const int orow = 16 * r1 + m;
                const bool ook = g2_row_ok<VL ? 2 : 0>(orow, olen);
                const size_t tok = (size_t)(otok + orow);
                const size_t obase = tok * rowT + (size_t)bh * kK + kBVW * vs + 16 * v1a;
                bf16* st_ = oStage[wv];
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    reinterpret_cast<short*>(st_)[m * 16 + q4 + 4 * e] = bf_rne(o[e]);
                r1a_o_wave_order();
                const int m2 = l & 15, g2_ = l >> 4;
                const v4bh ov = *reinterpret_cast<const v4bh*>(
                        reinterpret_cast<const short*>(st_) + m2 * 16 + 4 * g2_);
                if (ook) *reinterpret_cast<v4bh*>(p.Ob + obase + 4 * g2_) = ov;
            }
        }
        // ---- GEMM2 for both K-blocks
        #pragma unroll
        for (int vt = 0; vt < NVT; ++vt) {
            f4 upd = f4{0.f,0.f,0.f,0.f};
            #pragma unroll
            for (int rt = 0; rt < 4; ++rt) {
                const v4bh va = ld8(vB + (16 * vt + m) * kPadT + 16 * rt + q4 * 4);
                upd = mmac(va, kf0[rt], upd);
            }
            ht0[vt] += upd;
        }
        #pragma unroll
        for (int vt = 0; vt < NVT; ++vt) {
            f4 upd = f4{0.f,0.f,0.f,0.f};
            #pragma unroll
            for (int rt = 0; rt < 4; ++rt) {
                const v4bh va = ld8(vB + (16 * vt + m) * kPadT + 16 * rt + q4 * 4);
                upd = mmac(va, kf1[rt], upd);
            }
            ht1[vt] += upd;
        }
        if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);
        loadK(nxt, kb0, kf0); loadK(nxt, kb1, kf1);
        if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);
        write_h(kb0, ht0);
        write_h(kb1, ht1);
    }
}
'''

anchor = 'extern "C" int g2_fwd_h_v8o(const void* Wn, const void* Kn, const void* Un, const void* Gn,'
assert t.count(anchor) == 1
t = t.replace(anchor, KERNEL + "\n" + anchor)

old = """    auto v8o4 = [&](auto vl) {"""
assert t.count(old) == 1
t = t.replace(old, """    // kra P-23: 64-column slabs (2 CTAs per head) at deep saturation, production modes only
    int wide_min = 256;
    if (const char* e = getenv("KDA_G2_V8W_MIN_CTAS")) wide_min = atoi(e);
    if (r1a && (mode == 1 || mode == 4) && 4L * nseg * H >= wide_min) {
        const dim3 gridw(kK / 64, (unsigned)nseg * H);
        auto v8w = [&](auto vl) {
            constexpr bool VL = decltype(vl)::value;
            if (mode == 1) hipLaunchKernelGGL((g2_fwd_h_v8w_kernel<VL, 1>), gridw, dim3(256), 0, s, p);
            else           hipLaunchKernelGGL((g2_fwd_h_v8w_kernel<VL, 4>), gridw, dim3(256), 0, s, p);
        };
        if (ctok != nullptr) v8w(std::true_type{});
        else                 v8w(std::false_type{});
        return (int)hipGetLastError();
    }
    auto v8o4 = [&](auto vl) {""")
P.write_text(t)
print("P-23 APPLIED")
