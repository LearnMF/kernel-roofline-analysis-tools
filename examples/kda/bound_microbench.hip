// kda_bound validation microbenchmarks (gfx936), built against the hip_kda G2 helpers so the MMAC /
// LDS instruction forms are exactly the production ones.
//   k_mmac_tile<MODE>: attainable MMAC throughput for KDA's tile patterns, 4 waves per CTA, operands
//     MODE 0 registers only | 1 A regs + B from LDS (the GEMM1 pattern: W in registers, h in LDS)
//     | 2 A and B from LDS (one ds_read_b64 per operand per MMAC)
//   k_step<O>: one forward-recurrence step per loop iteration with the v8o4 dependency structure
//     (GEMM1 W h -> v_new = U - acc -> LDS (transposed) -> barrier -> GEMM2 k^T v_new -> state update
//     -> h (bf16) -> LDS -> barrier), no global memory in the loop; O = 1 adds the o GEMMs.
// hipcc -O3 --offload-arch=gfx936 -I<g2> bound_microbench.hip -o bmb && ./bmb
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include "g2_common.h"

using namespace g2;

template <int MODE>
__global__ void __launch_bounds__(256) k_mmac_tile(int iters, float* out) {
    __shared__ __align__(16) bf16 As[64 * kPadK];
    __shared__ __align__(16) bf16 Bs[kBV * kPadK];
    const int tid = threadIdx.x, l = tid & 63, m = l & 15, q4 = l >> 4;
    for (int i = tid; i < 64 * kPadK; i += 256) reinterpret_cast<short*>(As)[i] = (short)(0x3c00 + (i & 7));
    for (int i = tid; i < kBV * kPadK; i += 256) reinterpret_cast<short*>(Bs)[i] = (short)(0x3c00 + (i & 3));
    __syncthreads();
    v4bh a[8];
    #pragma unroll
    for (int kt = 0; kt < 8; ++kt) a[kt] = ld8(As + m * kPadK + 16 * kt + q4 * 4);
    f4 acc0 = {0.f, 0.f, 0.f, 0.f}, acc1 = acc0, acc2 = acc0, acc3 = acc0;
    for (int it = 0; it < iters; ++it) {
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
            v4bh av = a[kt];
            if (MODE == 2) av = ld8(As + ((it + kt) & 63) * kPadK + 16 * kt + q4 * 4);
            v4bh b0 = av, b1 = av;
            if (MODE >= 1) {
                b0 = ld8(Bs + (0 * 16 + m) * kPadK + 16 * kt + q4 * 4);
                b1 = ld8(Bs + (1 * 16 + m) * kPadK + 16 * kt + q4 * 4);
            }
            if (kt & 1) { acc2 = mmac(av, b0, acc2); acc3 = mmac(av, b1, acc3); }
            else        { acc0 = mmac(av, b0, acc0); acc1 = mmac(av, b1, acc1); }
        }
    }
    const f4 s = (acc0 + acc1) + (acc2 + acc3);
    out[blockIdx.x * 256 + tid] = s[0] + s[1] + s[2] + s[3];
}

template <int O>
__global__ void __launch_bounds__(256) k_step(int NT, float* out) {
    __shared__ __align__(16) bf16 hB[kBV * kPadK];
    __shared__ __align__(16) bf16 vB[kBV * kPadT];
    const int tid = threadIdx.x, wv = tid >> 6, l = tid & 63, m = l & 15, q4 = l >> 4;
    const int r1 = wv, kb0 = wv, kb1 = wv + 4;
    v4bh wf[8], kf0[4], kf1[4], qgf[8], aqf[4], uf0, uf1;
    #pragma unroll
    for (int j = 0; j < 8; ++j) { wf[j] = v4bh{(short)(0x2c00 + j), (short)(0x2c01), (short)0x2c02, (short)(0x2c03 + l)}; qgf[j] = wf[j]; }
    #pragma unroll
    for (int j = 0; j < 4; ++j) { kf0[j] = v4bh{(short)(0x2800 + j), (short)0x2801, (short)0x2802, (short)(0x2803 + m)}; kf1[j] = kf0[j]; aqf[j] = kf0[j]; }
    uf0 = v4bh{(short)0x3c00, (short)0x3c00, (short)0x3c00, (short)0x3c00}; uf1 = uf0;
    f4 ht0[2], ht1[2];
    ht0[0] = f4{0.f, 0.f, 0.f, 0.f}; ht0[1] = ht0[0]; ht1[0] = ht0[0]; ht1[1] = ht0[0];
    const f4 d0 = {0.999f, 0.998f, 0.997f, 0.996f};
    auto write_h = [&](int kb, f4* ht) {
        #pragma unroll
        for (int vt = 0; vt < 2; ++vt)
            #pragma unroll
            for (int e = 0; e < 4; ++e)
                reinterpret_cast<short*>(hB)[(16 * vt + m) * kPadK + 16 * kb + c_col(l, e)] = bf_rne(ht[vt][e]);
    };
    write_h(kb0, ht0); write_h(kb1, ht1);
    float osum = 0.f;
    for (int i = 0; i < NT; ++i) {
        lds_barrier();                                          // hB (h_i) ready
        f4 acc0 = {0.f, 0.f, 0.f, 0.f}, acc1 = acc0;
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
            acc0 = mmac(wf[kt], ld8(hB + (0 * 16 + m) * kPadK + 16 * kt + q4 * 4), acc0);
            acc1 = mmac(wf[kt], ld8(hB + (1 * 16 + m) * kPadK + 16 * kt + q4 * 4), acc1);
        }
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float u0 = __uint_as_float(((uint32_t)(unsigned short)uf0[e]) << 16);
            reinterpret_cast<short*>(vB)[(16 * 0 + c_col(l, e)) * kPadT + 16 * r1 + m] = bf_rne(u0 - acc0[e]);
            reinterpret_cast<short*>(vB)[(16 * 1 + c_col(l, e)) * kPadT + 16 * r1 + m] = bf_rne(u0 - acc1[e]);
        }
        ht0[0] *= d0; ht0[1] *= d0; ht1[0] *= d0; ht1[1] *= d0;
        f4 oh0 = {0.f, 0.f, 0.f, 0.f}, oh1 = oh0;
        if (O) {
            #pragma unroll
            for (int kt = 0; kt < 8; ++kt) {
                oh0 = mmac(qgf[kt], ld8(hB + (0 * 16 + m) * kPadK + 16 * kt + q4 * 4), oh0);
                oh1 = mmac(qgf[kt], ld8(hB + (1 * 16 + m) * kPadK + 16 * kt + q4 * 4), oh1);
            }
        }
        lds_barrier();                                          // vB (v_new_i) ready
        if (O) {
            #pragma unroll
            for (int v1a = 0; v1a < 2; ++v1a) {
                f4 oa = {0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int st = 0; st < 4; ++st) oa = mmac(aqf[st], ld8(vB + (16 * v1a + m) * kPadT + 16 * st + q4 * 4), oa);
                const f4& oh = v1a ? oh1 : oh0;
                osum += oh[0] * 0.088f + oa[0] + oh[3] + oa[3];
            }
        }
        #pragma unroll
        for (int vt = 0; vt < 2; ++vt) {
            f4 upd = {0.f, 0.f, 0.f, 0.f}, upd1 = upd;
            #pragma unroll
            for (int rt = 0; rt < 4; ++rt) {
                const v4bh va = ld8(vB + (16 * vt + m) * kPadT + 16 * rt + q4 * 4);
                upd = mmac(va, kf0[rt], upd);
                upd1 = mmac(va, kf1[rt], upd1);
            }
            ht0[vt] += upd; ht1[vt] += upd1;
        }
        write_h(kb0, ht0); write_h(kb1, ht1);
    }
    const f4 s = ht0[0] + ht1[1];
    out[blockIdx.x * 256 + tid] = s[0] + s[1] + s[2] + s[3] + osum;
}

template <typename K>
static float time_ms(K kern, int grid, int arg, float* out, int reps = 5) {
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    hipLaunchKernelGGL(kern, dim3(grid), dim3(256), 0, 0, arg, out);
    hipDeviceSynchronize();
    float best = 1e30f;
    for (int r = 0; r < reps; ++r) {
        hipEventRecord(a);
        hipLaunchKernelGGL(kern, dim3(grid), dim3(256), 0, 0, arg, out);
        hipEventRecord(b); hipEventSynchronize(b);
        float ms; hipEventElapsedTime(&ms, a, b); best = ms < best ? ms : best;
    }
    return best;
}

int main() {
    float* out; hipMalloc(&out, 1 << 24);
    const int iters = 20000;
    for (int ctas_per_cu : {1, 2, 4}) {
        const int grid = 80 * ctas_per_cu;
        const double flop = 2.0 * grid * 4 * (double)iters * 16 * 4096;   // 16 MMAC 16x16x16 per wave per iter
        const float t0 = time_ms(k_mmac_tile<0>, grid, iters, out);
        const float t1 = time_ms(k_mmac_tile<1>, grid, iters, out);
        const float t2 = time_ms(k_mmac_tile<2>, grid, iters, out);
        printf("mmac tile, %d CTA/CU: regs %.1f TF | A regs + B LDS %.1f TF | A,B LDS %.1f TF\n", ctas_per_cu,
               flop / t0 / 1e9, flop / t1 / 1e9, flop / t2 / 1e9);
    }
    for (int NT : {256, 1024}) {
        for (int grid : {48, 80, 160, 192, 384}) {
            const float a = time_ms(k_step<0>, grid, NT, out), b = time_ms(k_step<1>, grid, NT, out);
            printf("recurrence step, NT=%d, %3d CTAs: state-only %.3f us/step | with o %.3f us/step\n", NT, grid,
                   a * 1e3 / NT, b * 1e3 / NT);
        }
    }
    return 0;
}
