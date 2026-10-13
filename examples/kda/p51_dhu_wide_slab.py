"""kra P-51 (A-class, dhu): a V-slab of NVH*16 columns per CTA (NVH = 4: 64 columns, 2 CTAs per
head) for wide-head long-sequence launches, the dhu counterpart of P-49.

Each of a head's 4 V-slab CTAs loads the same per-step K-side tiles (KgA, Qg, Wb, Gn); with 64
columns per CTA they are issued by 2 CTAs instead of 4.  Every V column's arithmetic (GEMM-A tile,
dv2 rounding, decay, GEMM-B tiles, aN/bN emits) is the same instruction sequence on the same
operands -> bit-identical.  Gated by KDA_G2_DHUW_MIN_H (default 40) and KDA_G2_DHUW_MIN_NT
(default 512) on the 4-wave route; KDA_G2_DHUW=0/1 forces it off/on (for the A/B).
    python3 p51_dhu_wide_slab.py <tree>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_engine.cuh"
t = P.read_text()

kern = r'''
// kra P-51: g2_dhu_v8b4_kernel with NVH 16-column V-tiles per CTA (NVH = 4: 64 columns, 2 CTAs/head)
template <int MODE = 0, bool LIN = false, int NVH = 4>
__global__ void __launch_bounds__(256) g2_dhu_v8b4w_kernel(BwdNParams p) {
    const bool hasBN = MODE == 0 ? p.HbN != nullptr : true;
    const bool hasDH = MODE == 0 ? p.dh != nullptr : false;
    const bool hasDV2 = MODE == 0 ? p.dv2 != nullptr : MODE == 2;
    const bool hasAN2 = MODE == 0 ? p.ANv2 != nullptr : MODE == 1;
    constexpr int BVW = 16 * NVH;
    __shared__ __align__(16) bf16 dhB[BVW * kPadK];
    __shared__ __align__(16) bf16 vB[BVW * kPadT];
    __shared__ __align__(16) v4bh dOs[2][NVH * 4 * 64];
    const int tid = threadIdx.x, wv = tid >> 6, l = tid & 63;
    const int vs = blockIdx.x;
    const int sg = blockIdx.y / p.H, bh = blockIdx.y - sg * p.H;
    const int c0 = p.seg ? p.seg[sg] : 0, c1 = p.seg ? p.seg[sg + 1] : p.NT;
    if (c0 >= c1) return;
    const size_t chunk = (size_t)bh * p.NT;
    const int r1 = wv, kb0 = wv, kb1 = wv + 4;
    f4 st0[NVH], st1[NVH];
    #pragma unroll
    for (int v = 0; v < NVH; ++v) { st0[v] = f4{0.f, 0.f, 0.f, 0.f}; st1[v] = st0[v]; }
    v4bh ka[8], qb0[4], wb0[4], qb1[4], wb1[4], dof[NVH], dvf[NVH];
    f4 gl0, gl1;
    auto loadKA = [&](int i) {
        const bf16* src = p.KgA + (((chunk + i) * 4 + r1) * 8) * 256 + l * 8;
        #pragma unroll
        for (int j = 0; j < 4; ++j) ld16x2(src + j * 512, ka[2 * j], ka[2 * j + 1]);
    };
    auto loadDV = [&](int i, int v1, v4bh& d) {
        d = ld8(p.dv + (((chunk + i) * 4 + r1) * 8 + NVH * vs + v1) * 256 + l * 4);
    };
    auto loadG = [&](int i, int kb, f4& gl) {
        gl = *reinterpret_cast<const f4*>(p.Gn + ((chunk + i) * 8 + kb) * 16 + (l >> 4) * 4);
    };
    auto loadQW = [&](int i, int kb, v4bh* qb, v4bh* wb) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            ld16x2(p.Qg + (((chunk + i) * 8 + kb) * 4) * 256 + j * 512 + l * 8, qb[2 * j], qb[2 * j + 1]);
            ld16x2(p.Wb + (((chunk + i) * 8 + kb) * 4) * 256 + j * 512 + l * 8, wb[2 * j], wb[2 * j + 1]);
        }
    };
    auto loadDO = [&](int i, int v1, v4bh& d) {
        d = ld8(p.dOT + (((chunk + i) * 8 + NVH * vs + v1) * 4 + r1) * 256 + l * 4);
    };
    v4bh hx0[NVH], hx1[NVH];
    auto write_dh = [&](int kb, f4* st, v4bh* hx) {
        #pragma unroll
        for (int vt = 0; vt < NVH; ++vt)
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                hx[vt][e] = bf_rne(st[vt][e]);
                reinterpret_cast<short*>(dhB)[(16 * vt + c_row(l)) * kPadK + 16 * kb + c_col(l, e)] = hx[vt][e];
            }
    };
    auto bN_emit = [&](int i, int kb) {
        if (hasBN) {
            const int q4_ = l >> 4, m_ = l & 15;
            #pragma unroll
            for (int vt = 0; vt < NVH; ++vt) {
                v4bh val;
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    val[e] = reinterpret_cast<const short*>(dhB)[(16 * vt + 4 * q4_ + e) * kPadK + 16 * kb + pi16(m_)];
                const size_t off = ((((size_t)i * p.H + bh) * 8 + kb) * 8 + (NVH * vs + vt)) * 256 + l * 4;
                *const_cast<v4bh*>(reinterpret_cast<const v4bh*>(p.HbN + off)) = val;
            }
        }
    };
    const int last = c1 - 1;
    loadKA(last);
    #pragma unroll
    for (int v = 0; v < NVH; ++v) loadDV(last, v, dvf[v]);
    loadG(last, kb0, gl0); loadG(last, kb1, gl1);
    loadQW(last, kb0, qb0, wb0); loadQW(last, kb1, qb1, wb1);
    #pragma unroll
    for (int v = 0; v < NVH; ++v) loadDO(last, v, dof[v]);
    write_dh(kb0, st0, hx0); write_dh(kb1, st1, hx1);
    {
        const int i = last;
        lds_barrier();
        bN_emit(i, kb0);
        bN_emit(i, kb1);
    }
    for (int s = 0; s < c1 - c0; ++s) {
        const int i = last - s, nxt = i > c0 ? i - 1 : c0;
        if (hasDH) {
            #pragma unroll
            for (int vt = 0; vt < NVH; ++vt) {
                *reinterpret_cast<v4bh*>(p.dh + (((chunk + i) * 8 + NVH * vs + vt) * 8 + kb0) * 256 + l * 4) = hx0[vt];
                *reinterpret_cast<v4bh*>(p.dh + (((chunk + i) * 8 + NVH * vs + vt) * 8 + kb1) * 256 + l * 4) = hx1[vt];
            }
        }
        #pragma unroll
        for (int v = 0; v < NVH; ++v) dOs[s & 1][(v * 4 + wv) * 64 + l] = dof[v];   // slot (v1, r1)
        #pragma unroll
        for (int v = 0; v < NVH; ++v) loadDO(nxt, v, dof[v]);
        lds_barrier();
        bN_emit(i, kb0);
        bN_emit(i, kb1);
        f4 acc[NVH];
        #pragma unroll
        for (int v = 0; v < NVH; ++v) acc[v] = f4{0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt)
            #pragma unroll
            for (int v = 0; v < NVH; ++v)
                acc[v] = mmac(ka[kt], ld8(dhB + (16 * v + (l & 15)) * kPadK + 16 * kt + (l >> 4) * 4), acc[v]);
        loadKA(nxt);
        #pragma unroll
        for (int v1 = 0; v1 < NVH; ++v1) {
            v4bh x;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float d = acc[v1][e] + __uint_as_float(((uint32_t)(unsigned short)dvf[v1][e]) << 16);
                x[e] = bf_rne(d);
                reinterpret_cast<short*>(vB)[(16 * v1 + c_col(l, e)) * kPadT + 16 * r1 + c_row(l)] = x[e];
            }
            if (hasDV2)
                *reinterpret_cast<v4bh*>(p.dv2 + (((chunk + i) * 4 + r1) * 8 + NVH * vs + v1) * 256 + l * 4) = x;
            if (hasAN2 && !LIN)
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    reinterpret_cast<short*>(p.ANv2)[((size_t)i * p.H + bh) * 8192 + (8 * r1 + NVH * vs + v1) * 256
                           + (l >> 4) + 4 * (l & 15) + 64 * e] = x[e];
        }
        #pragma unroll
        for (int v = 0; v < NVH; ++v) loadDV(nxt, v, dvf[v]);
        {
            f4 d0, d1;
            #pragma unroll
            for (int e = 0; e < 4; ++e) { d0[e] = exp2f(gl0[e]); d1[e] = exp2f(gl1[e]); }
            #pragma unroll
            for (int v = 0; v < NVH; ++v) { st0[v] *= d0; st1[v] *= d1; }
        }
        loadG(nxt, kb0, gl0); loadG(nxt, kb1, gl1);
        lds_barrier();
        if (hasAN2 && LIN) {
            #pragma unroll
            for (int v1g = 0; v1g < NVH; ++v1g) {
                v4bh an;
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    an[e] = reinterpret_cast<const short*>(vB)[(16 * v1g + 4 * (l >> 4) + e) * kPadT + 16 * r1 + (l & 15)];
                *reinterpret_cast<v4bh*>(p.ANv2 + ((size_t)i * p.H + bh) * 8192 + (8 * r1 + NVH * vs + v1g) * 256 + l * 4) = an;
            }
        }
        #pragma unroll
        for (int vt = 0; vt < NVH; ++vt) {
            f4 qd = f4{0.f, 0.f, 0.f, 0.f}, wd = f4{0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int rt = 0; rt < 4; ++rt) {
                qd = mmac(dOs[s & 1][(vt * 4 + rt) * 64 + l], qb0[rt], qd);
                wd = mmac(ld8(vB + (16 * vt + (l & 15)) * kPadT + 16 * rt + (l >> 4) * 4), wb0[rt], wd);
            }
            st0[vt] += qd * p.scale - wd;
        }
        #pragma unroll
        for (int vt = 0; vt < NVH; ++vt) {
            f4 qd = f4{0.f, 0.f, 0.f, 0.f}, wd = f4{0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int rt = 0; rt < 4; ++rt) {
                qd = mmac(dOs[s & 1][(vt * 4 + rt) * 64 + l], qb1[rt], qd);
                wd = mmac(ld8(vB + (16 * vt + (l & 15)) * kPadT + 16 * rt + (l >> 4) * 4), wb1[rt], wd);
            }
            st1[vt] += qd * p.scale - wd;
        }
        loadQW(nxt, kb0, qb0, wb0); loadQW(nxt, kb1, qb1, wb1);
        write_dh(kb0, st0, hx0);
        write_dh(kb1, st1, hx1);
    }
}
'''
anchor = 'extern "C" int g2_dhu_w8b(const void* KgA, const void* Qg, const void* Wb, const void* dOT, const void* dv,'
assert t.count(anchor) == 1
t = t.replace(anchor, kern + "\n" + anchor)

old = """    if (dhu4) {
        // v8b4: the compile-time instance measured 8% SLOWER (kra P-10) -> generic only"""
new = """    // kra P-51: wide-head long-sequence launches use 64-column V-slabs (2 CTAs per head) on the
    // 4-wave route: the K-side tiles of a step are issued by 2 CTAs instead of 4 (bit-identical)
    int dhuw = dhu4 && H >= g2_env_long("KDA_G2_DHUW_MIN_H", 40) && T / kBT >= g2_env_long("KDA_G2_DHUW_MIN_NT", 512);
    if (const char* e = getenv("KDA_G2_DHUW")) dhuw = atoi(e) != 0 && dhu4;
    if (dhuw) {
        const dim3 gridw(kK / 64, (unsigned)nseg * H);
        bool linw = 2L * nseg * H <= 96;
        if (const char* e = getenv("KDA_G2_DHU_LIN")) linw = atoi(e) != 0;
        if (linw) hipLaunchKernelGGL((g2_dhu_v8b4w_kernel<0, true>), gridw, dim3(256), 0, s, p);
        else      hipLaunchKernelGGL((g2_dhu_v8b4w_kernel<0, false>), gridw, dim3(256), 0, s, p);
        return (int)hipGetLastError();
    }
    if (dhu4) {
        // v8b4: the compile-time instance measured 8% SLOWER (kra P-10) -> generic only"""
assert t.count(old) == 1
t = t.replace(old, new)
P.write_text(t)
print("P-51 APPLIED")
