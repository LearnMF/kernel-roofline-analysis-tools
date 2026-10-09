// SQTT bench: runs the PRODUCTION G2 kernels (via their launchers) back to back
// so xprof --enable-sqtt can capture each one. Data is random fill; only timing/
// instruction behavior matters.
// Build: cd <tree>/csrc/g2 && hipcc -O3 -std=c++17 --offload-arch=gfx936 -I. \
//   -DHIP_ENABLE_WARP_SYNC_BUILTINS=1 bn_sqtt.hip -o bn_sqtt
// Run:   bn_sqtt T H reps
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "g2_common.h"
#include "g2_fwd_intra.cuh"
#include "g2_fwd_intra2.cuh"
#include "g2_fwd_prep.cuh"
#include "g2_fwd_v8.cuh"
#include "g2_engine.cuh"
#include "g2_wy.cuh"
#include "g2_wu.cuh"
#include "g2_dav.cuh"
#include "g2_bwd_tail.cuh"

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP %s @%d: %s\n", #x, __LINE__, hipGetErrorString(e_)); exit(1); } } while (0)
#define CKR(x) do { int e_ = (x); if (e_ != 0) { \
    fprintf(stderr, "RC %s @%d: %d\n", #x, __LINE__, e_); exit(1); } } while (0)

__device__ __forceinline__ uint32_t hsh(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33;
    return (uint32_t)x;
}
__global__ void fill_bf16(short* p, size_t n, uint64_t seed, float amp) {
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += 256ull * gridDim.x) {
        const float u = (hsh(seed * 0x9e3779b97f4a7c15ull + i) >> 8) * (1.f / 16777216.f) * 2.f - 1.f;
        p[i] = g2::bf_rne(u * amp);
    }
}
__global__ void fill_f32(float* p, size_t n, uint64_t seed, float lo, float hi) {
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += 256ull * gridDim.x) {
        const float u = (hsh(seed * 0x9e3779b97f4a7c15ull + i) >> 8) * (1.f / 16777216.f);
        p[i] = lo + (hi - lo) * u;
    }
}
__global__ void fill_g(float* g, int T, int H) {   // per-chunk decreasing gate (log2-domain)
    const size_t n = (size_t)T * H * 128;
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += 256ull * gridDim.x) {
        const int t = (int)(i / ((size_t)H * 128));
        const size_t hc = i % ((size_t)H * 128);
        const float step = 0.02f + 0.18f * ((hsh(777 + hc) >> 8) * (1.f / 16777216.f));
        g[i] = -(float)(t % 64 + 1) * step;
    }
}
__global__ void fill_beta(float* b, size_t n) {   // beta in (0,1)
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += 256ull * gridDim.x)
        b[i] = 0.2f + 0.6f * ((hsh(999 + i) >> 8) * (1.f / 16777216.f));
}
__global__ void spin_clock(unsigned long long n) {   // delimit the measured window
    const unsigned long long t0 = clock64();
    while (clock64() - t0 < n) {}
}

int main(int argc, char** argv) {
    const int T = argc > 1 ? atoi(argv[1]) : 8192, H = argc > 2 ? atoi(argv[2]) : 12;
    const int reps = argc > 3 ? atoi(argv[3]) : 3, NT = T / 64;
    const size_t NROW = (size_t)NT * H * 8192;      // bf16 "row" tensor (also fits T*H*128)
    const size_t NST = (size_t)NT * H * 16384;      // state-sized bf16 (h / HbN)
    const size_t NA = (size_t)NT * H * 4096;        // 64-wide A-native
    const size_t NF = (size_t)NT * H * 8192;        // f32 generic
    auto dmal = [](size_t bytes) { void* p; CK(hipMalloc(&p, bytes)); return p; };

    short *q = (short*)dmal(NROW * 2), *k = (short*)dmal(NROW * 2), *v = (short*)dmal(NROW * 2),
          *Wn = (short*)dmal(NROW * 2), *Kn = (short*)dmal(NROW * 2), *Un = (short*)dmal(NROW * 2),
          *QgA = (short*)dmal(NROW * 2), *AqkA = (short*)dmal(NA * 2), *Aqk = (short*)dmal(NA * 2),
          *Akk = (short*)dmal(NA * 2), *Kb = (short*)dmal(NROW * 2), *Ob = (short*)dmal(NROW * 2),
          *Vn = (short*)dmal(NROW * 2), *HbN = (short*)dmal(NST * 2), *ANv = (short*)dmal(NROW * 2),
          *kgB = (short*)dmal(NROW * 2), *vB = (short*)dmal(NROW * 2), *dv = (short*)dmal(NROW * 2),
          *dO = (short*)dmal(NROW * 2), *dh = (short*)dmal(NST * 2), *dv2 = (short*)dmal(NROW * 2),
          *dOT = (short*)dmal(NROW * 2), *Wb = (short*)dmal(NROW * 2), *Qg = (short*)dmal(NROW * 2),
          *KgA = (short*)dmal(NROW * 2), *dwT = (short*)dmal(NROW * 2), *ANv2 = (short*)dmal(NROW * 2);
    float *g = (float*)dmal(NF * 4), *beta = (float*)dmal(NF * 4), *A_log = (float*)dmal(NF * 4),
          *dt_bias = (float*)dmal(NF * 4), *q_rstd = (float*)dmal(NF * 4), *k_rstd = (float*)dmal(NF * 4),
          *Gn = (float*)dmal(NF * 4), *R = (float*)dmal(NF * 4), *dAqk = (float*)dmal(NF * 4),
          *dq = (float*)dmal(NF * 4), *dk = (float*)dmal(NF * 4), *dg = (float*)dmal(NF * 4),
          *db = (float*)dmal(NF * 4), *dA = (float*)dmal(NF * 4), *db_part = (float*)dmal(NF * 4),
          *dA_log = (float*)dmal(NF * 4), *ddt = (float*)dmal(NF * 4), *dbO = (float*)dmal(NF * 4);
    short* all16[] = {q, k, v, Wn, Kn, Un, QgA, AqkA, Aqk, Akk, Kb, Vn, HbN, ANv, kgB, vB, dv,
                      dO, dh, dv2, dOT, Wb, Qg, KgA, dwT, ANv2};
    for (int j = 0; j < 26; ++j)
        hipLaunchKernelGGL(fill_bf16, dim3(2048), dim3(256), 0, 0, all16[j],
                           (j == 12 || j == 18) ? NST : (j == 7 || j == 8 || j == 9) ? NA : NROW,
                           11 + j, 0.1f);
    for (float* f : {g, beta, A_log, dt_bias, q_rstd, k_rstd, Gn, R, dAqk, dq, dk, dg, db, dA, db_part})
        fill_f32<<<2048, 256>>>(f, NF, 43, -0.5f, 0.5f);
    fill_beta<<<2048, 256>>>(beta, NF);
    CK(hipDeviceSynchronize());
    const float scale = 0.088388f;   // 1/sqrt(128)

    for (int it = 0; it < reps; ++it) {
        CKR(g2_fwd_prep(q, k, v, g, A_log, dt_bias, beta, Wn, Kn, Un, QgA, AqkA, Aqk, Akk,
                        q_rstd, k_rstd, Gn, R, Kb, -5.0, 1.0, /*gate*/0, scale, H, NT,
                        nullptr, nullptr, 0));
        // forward recurrence with bf16-o emit (production fwd; also the ckpt recompute path
        // adds HbN/ANv -- capture that variant too)
        CKR(g2::g2_fwd_h_v8o(Wn, Kn, Un, Gn, QgA, AqkA, nullptr, Ob, Vn, nullptr, nullptr,
                            scale, 1, T, H, nullptr, 1, nullptr, nullptr, 0));
        CKR(g2::g2_fwd_h_v8o(Wn, Kn, Un, Gn, QgA, AqkA, nullptr, Ob, Vn, HbN, ANv,
                            scale, 1, T, H, nullptr, 1, nullptr, nullptr, 0));
        CKR(g2_wu(k, v, q, beta, Aqk, g, Wn, Wb, Un, KgA, Kn, QgA, kgB, vB,
                     1, T, H, nullptr, nullptr, 0));
        CKR(g2_dav(dO, Vn, Aqk, dA, dv2, dOT, ANv2, scale, 1, T, H,
                      nullptr, nullptr, 0));
        CKR(g2::g2_dhu_v8b(KgA, Qg, Wb, dOT, dv, Gn, dv2, HbN, ANv2, scale, 1, T, H,
                          nullptr, 1, 0));
        CKR(g2::g2_wy_s4_split(kgB, k, vB, Vn, g, beta, Aqk, HbN, dO, dh, dv, dAqk,
                              dq, dk, dv2, dg, db, dA, dwT, db_part, scale, T, H, NT,
                              q, v, nullptr, nullptr, 0));
        CKR(g2_bwd_tail(dq, dk, dg, db, q, k, q_rstd, k_rstd, g, A_log, dt_bias,
                           beta, Ob, dv2, ANv2, dA_log, ddt, (short*)dbO,
                           dA, db_part, -5.0, 1.0, /*use_l2*/1, /*use_beta*/1,
                           1, T, H, nullptr, nullptr, NT, /*gate_none*/0, /*beta_f32*/1, 0));
        spin_clock<<<1, 32>>>(3000000);
        CK(hipDeviceSynchronize());
    }
    printf("BN_DONE T=%d H=%d reps=%d\n", T, H, reps);
    return 0;
}
