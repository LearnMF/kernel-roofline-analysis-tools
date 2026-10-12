"""kra P-49 (A-class, forward / recompute recurrence): a V-slab of 64 columns per CTA (2 CTAs per head)
for wide-head launches (H >= KDA_G2_V8O4W_MIN_H, default 40 = TP2).

Every V-slab CTA of a head loads the same per-step K-side tiles (W, K, QgA, AqkA, G: 14 of the 17
loads per wave per step; only U is slab-specific).  At TP2 the recurrence kernels are VMEM-issue bound
(TA 72-73%) and re-fetch those tiles (executed bytes 1.24-1.31x the interface): with 64 columns per CTA
the shared loads are issued by 2 CTAs instead of 4 (~-47% VMEM instructions).  Each V column's
arithmetic (GEMM1 tile, v_new, decay, GEMM2 tile, o tile) is the same instruction sequence on the
same operands -> bit-identical.  TP8 / TP4 keep the 32-column kernel (parallelism).
python3 p49_rec_wide_slab.py <tree>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_fwd_v8.cuh"
t = P.read_text()

kern = r'''
// kra P-49: g2_fwd_h_v8o4_kernel with NVH 16-column V-tiles per CTA (NVH = 4: 64 columns, 2 CTAs/head)
template <bool VL, int MODE = 0, int NVH = 4>
__global__ void __launch_bounds__(256) g2_fwd_h_v8o4w_kernel(FwdOParams p) {
    const bool hasO  = MODE == 0 ? p.O != nullptr : false;
    const bool hasOb = MODE == 0 ? p.Ob != nullptr : (MODE != 3 && MODE != 4);
    const bool hasBN = MODE == 0 ? p.HbN != nullptr : MODE != 1;
    const bool hasVn = MODE == 0 ? p.Vn != nullptr : MODE != 1;
    const bool hasAN = MODE == 0 ? p.ANv != nullptr : (MODE != 1 && MODE != 4);
    constexpr int BVW = 16 * NVH;
    __shared__ __align__(16) bf16 hB[BVW * kPadK];
    __shared__ __align__(16) bf16 vB[BVW * kPadT];
    __shared__ __align__(16) bf16 oStage[4][256];
    const int tid = threadIdx.x, wv = tid >> 6, l = tid & 63;
    const int vs = blockIdx.x;
    const int sg = blockIdx.y / p.H, bh = blockIdx.y - sg * p.H;
    const int c0 = p.seg ? p.seg[sg] : 0, c1 = p.seg ? p.seg[sg + 1] : p.NT;
    if (c0 >= c1) return;
    const size_t chunk = (size_t)bh * p.NT;
    const int r1 = wv;
    const int kb0 = wv, kb1 = wv + 4;
    const int m = l & 15, q4 = l >> 4;
    const size_t rowT = (size_t)p.H * kK;

    f4 ht0[NVH], ht1[NVH];
    #pragma unroll
    for (int v = 0; v < NVH; ++v) { ht0[v] = f4{0.f,0.f,0.f,0.f}; ht1[v] = ht0[v]; }
    v4bh wf[8], kf0[4], kf1[4], uf[NVH], qgf[8], aqf[4];
    f4 gl0, gl1;

    auto loadW = [&](int i) {
        const bf16* src = p.Wn + (((chunk + i) * 4 + r1) * 8) * 256 + l * 8;
        #pragma unroll
        for (int j = 0; j < 4; ++j) ld16x2(src + j * 512, wf[2 * j], wf[2 * j + 1]);
    };
    auto loadQg = [&](int i) {
        const bf16* src = p.QgA + (((chunk + i) * 4 + r1) * 8) * 256 + l * 8;
        #pragma unroll
        for (int j = 0; j < 4; ++j) ld16x2(src + j * 512, qgf[2 * j], qgf[2 * j + 1]);
    };
    auto loadAq = [&](int i) {
        const bf16* src = p.AqkA + (((chunk + i) * 4 + r1) * 4) * 256 + l * 8;
        #pragma unroll
        for (int j = 0; j < 2; ++j) ld16x2(src + j * 512, aqf[2 * j], aqf[2 * j + 1]);
    };
    auto loadU = [&](int i) {                         // paired V-tiles (NVH*vs + 2pr, +1)
        #pragma unroll
        for (int pr = 0; pr < NVH / 2; ++pr)
            ld16x2(p.Un + (((chunk + i) * 4 + r1) * 8 + NVH * vs + 2 * pr) * 256 + l * 8, uf[2 * pr], uf[2 * pr + 1]);
    };
    auto loadG = [&](int i, int kb, f4& gl) {
        gl = *reinterpret_cast<const f4*>(p.Gn + ((chunk + i) * 8 + kb) * 16 + q4 * 4);
    };
    auto loadK = [&](int i, int kb, v4bh* kf) {
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            ld16x2(p.Kn + (((chunk + i) * 8 + kb) * 4) * 256 + j * 512 + l * 8, kf[2 * j], kf[2 * j + 1]);
    };
    auto write_h = [&](int kb, f4* ht) {
        #pragma unroll
        for (int vt = 0; vt < NVH; ++vt)
            #pragma unroll
            for (int e = 0; e < 4; ++e)
                reinterpret_cast<short*>(hB)[(16 * vt + m) * kPadK + 16 * kb + c_col(l, e)] = bf_rne(ht[vt][e]);
    };
    auto bN_emit = [&](int i2, int kb) {
        if (hasBN) {
            #pragma unroll
            for (int vt = 0; vt < NVH; ++vt) {
                v4bh val;
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    val[e] = reinterpret_cast<const short*>(hB)[(16 * vt + 4 * q4 + e) * kPadK + 16 * kb + pi16(m)];
                const size_t off = ((((size_t)i2 * p.H + bh) * 8 + kb) * 8 + (NVH * vs + vt)) * 256 + l * 4;
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
        if (hasO || hasOb) {
            if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);
            loadQg(i); loadAq(i);
            if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);
        }
        bN_emit(i, kb0);
        bN_emit(i, kb1);
        f4 acc[NVH];
        #pragma unroll
        for (int v = 0; v < NVH; ++v) acc[v] = f4{0.f,0.f,0.f,0.f};
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt)
            #pragma unroll
            for (int v = 0; v < NVH; ++v)
                acc[v] = mmac(wf[kt], ld8(hB + (16 * v + m) * kPadK + 16 * kt + q4 * 4), acc[v]);
        if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);
        loadW(nxt);
        if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);
        #pragma unroll
        for (int v = 0; v < NVH; ++v) {
            v4bh x;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float u = __uint_as_float(((uint32_t)(unsigned short)uf[v][e]) << 16);
                x[e] = bf_rne(u - acc[v][e]);
                reinterpret_cast<short*>(vB)[(16 * v + c_col(l, e)) * kPadT + 16 * r1 + m] = x[e];
            }
            if (hasVn)
                *reinterpret_cast<v4bh*>(p.Vn + (((chunk + i) * 4 + r1) * 8 + NVH * vs + v) * 256 + l * 4) = x;
            if (hasAN)
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    reinterpret_cast<short*>(p.ANv)[((size_t)i * p.H + bh) * 8192 + (8 * r1 + NVH * vs + v) * 256
                           + (l >> 4) + 4 * (l & 15) + 64 * e] = x[e];
        }
        if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);
        loadU(nxt);
        if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);
        {
            f4 d0, d1;
            #pragma unroll
            for (int e = 0; e < 4; ++e) { d0[e] = exp2f(gl0[e]); d1[e] = exp2f(gl1[e]); }
            #pragma unroll
            for (int v = 0; v < NVH; ++v) { ht0[v] *= d0; ht1[v] *= d1; }
        }
        if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);
        loadG(nxt, kb0, gl0); loadG(nxt, kb1, gl1);
        if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);
        f4 oh[NVH];
        #pragma unroll
        for (int v = 0; v < NVH; ++v) oh[v] = f4{0.f,0.f,0.f,0.f};
        if (hasO || hasOb) {
            #pragma unroll
            for (int kt = 0; kt < 8; ++kt)
                #pragma unroll
                for (int v = 0; v < NVH; ++v)
                    oh[v] = mmac(qgf[kt], ld8(hB + (16 * v + m) * kPadK + 16 * kt + q4 * 4), oh[v]);
        }
        lds_barrier();                                   // vB (v_new_i) ready
        if (hasO || hasOb) {
            #pragma unroll
            for (int v1a = 0; v1a < NVH; ++v1a) {
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
                const size_t obase = tok * rowT + (size_t)bh * kK + BVW * vs + 16 * v1a;
                if (hasO) {
                    if (ook) {
                        #pragma unroll
                        for (int e = 0; e < 4; ++e)
                            p.O[obase + q4 + 4 * e] = o[e];
                    }
                } else {
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
        }
        #pragma unroll
        for (int vt = 0; vt < NVH; ++vt) {
            f4 upd = f4{0.f,0.f,0.f,0.f};
            #pragma unroll
            for (int rt = 0; rt < 4; ++rt) {
                const v4bh va = ld8(vB + (16 * vt + m) * kPadT + 16 * rt + q4 * 4);
                upd = mmac(va, kf0[rt], upd);
            }
            ht0[vt] += upd;
        }
        #pragma unroll
        for (int vt = 0; vt < NVH; ++vt) {
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
t = t.replace(anchor, kern + "\n" + anchor)

old = """    if (ctok != nullptr) {
        if (r1a) v8o4(std::true_type{});
        else     hipLaunchKernelGGL(g2_fwd_h_v8o_kernel<true>, grid, dim3(512), 0, s, p);
    } else {
        if (r1a) v8o4(std::false_type{});
        else     hipLaunchKernelGGL(g2_fwd_h_v8o_kernel<false>, grid, dim3(512), 0, s, p);
    }
    return (int)hipGetLastError();"""
new = """    // kra P-49: wide-head launches use 64-column V-slabs (2 CTAs per head): the K-side tiles of a step
    // are issued by 2 CTAs instead of 4 (bit-identical per column)
    // gated to long sequences: at NT < 512 the 96-CTA wide grid's CU imbalance cost more than the shared
    // loads it saves (16K/48 mode 4 0.91x, 8K/96 0.68x); 32K/48 1.02-1.12x, 64K/48 1.12-1.40x
    if (r1a && H >= g2_env_long("KDA_G2_V8O4W_MIN_H", 40) && T / kBT >= g2_env_long("KDA_G2_V8O4W_MIN_NT", 512)) {
        const dim3 gridw(kK / 64, (unsigned)nseg * H);
        auto v8o4w = [&](auto vl) {
            constexpr bool VL = decltype(vl)::value;
            switch (mode) {
                case 1: hipLaunchKernelGGL((g2_fwd_h_v8o4w_kernel<VL, 1>), gridw, dim3(256), 0, s, p); break;
                case 2: hipLaunchKernelGGL((g2_fwd_h_v8o4w_kernel<VL, 2>), gridw, dim3(256), 0, s, p); break;
                case 3: hipLaunchKernelGGL((g2_fwd_h_v8o4w_kernel<VL, 3>), gridw, dim3(256), 0, s, p); break;
                case 4: hipLaunchKernelGGL((g2_fwd_h_v8o4w_kernel<VL, 4>), gridw, dim3(256), 0, s, p); break;
                default: hipLaunchKernelGGL((g2_fwd_h_v8o4w_kernel<VL, 0>), gridw, dim3(256), 0, s, p); break;
            }
        };
        if (ctok != nullptr) v8o4w(std::true_type{}); else v8o4w(std::false_type{});
        return (int)hipGetLastError();
    }
    if (ctok != nullptr) {
        if (r1a) v8o4(std::true_type{});
        else     hipLaunchKernelGGL(g2_fwd_h_v8o_kernel<true>, grid, dim3(512), 0, s, p);
    } else {
        if (r1a) v8o4(std::false_type{});
        else     hipLaunchKernelGGL(g2_fwd_h_v8o_kernel<false>, grid, dim3(512), 0, s, p);
    }
    return (int)hipGetLastError();"""
assert t.count(old) == 1
t = t.replace(old, new)
P.write_text(t)
print("P-49 APPLIED")
