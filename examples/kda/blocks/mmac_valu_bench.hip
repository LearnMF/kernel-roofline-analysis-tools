// Building-block library P1: MMAC dependency / issue and VALU / transcendental / conversion costs on gfx936.
//   MMAC  : each wave keeps I independent accumulators and updates each one dependently N times
//           (bf16 16x16x16 and fp32 16x16x8, the two forms the KDA kernels use); waves / SIMD = 1, 2, 4.
//           cycles per MMAC per wave (I = 1 -> dependent latency).  MIX: + V independent v_fma per MMAC.
//   VALU  : exact instructions through inline asm; latency = one dependent chain in 1 wave / CU,
//           throughput = 8 independent chains x 4 waves / SIMD -> cycles per instruction per SIMD.
//   hipcc -O3 --offload-arch=gfx936 mmac_valu_bench.hip -o mvb && ./mvb > mmac_valu_bench.txt
#include <hip/hip_runtime.h>
#include <cstdio>

typedef short v4bh __attribute__((ext_vector_type(4)));
typedef float f4 __attribute__((ext_vector_type(4)));
typedef float f2v __attribute__((ext_vector_type(2)));

static double clk = 1.5e9;

template <int FORM, int I, int V>
__global__ void __launch_bounds__(256) k_mmac(int n, float* out) {
    f4 acc[I];
    #pragma unroll
    for (int i = 0; i < I; ++i) acc[i] = f4{0.f, (float)i, 0.f, 0.f};
    const v4bh a = v4bh{(short)threadIdx.x, 1, 2, 3}, b = v4bh{3, 2, 1, (short)blockIdx.x};
    const f2v fa = f2v{(float)threadIdx.x, 1.f}, fb = f2v{2.f, (float)blockIdx.x};
    float x[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = threadIdx.x + j;
    for (int it = 0; it < n; ++it) {
        #pragma unroll
        for (int i = 0; i < I; ++i) {
            if (FORM == 0) acc[i] = __builtin_amdgcn_mmac_f32_16x16x16bf16(a, b, acc[i]);
            else acc[i] = __builtin_amdgcn_mmac_f32_16x16x8f32(fa, fb, acc[i]);
            #pragma unroll
            for (int v = 0; v < V; ++v) asm volatile("v_fma_f32 %0, %0, %0, 1.0" : "+v"(x[v & 7]));
        }
    }
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < I; ++i) s += acc[i][0] + acc[i][3];
    #pragma unroll
    for (int j = 0; j < 8; ++j) s += x[j];
    if (s == 1.2345f) out[blockIdx.x * 256 + threadIdx.x] = s;
}

#define VOP(NAME, ASM)                                                                                       \
    template <int CH>                                                                                        \
    __global__ void __launch_bounds__(256) k_##NAME(int n, float* out) {                                     \
        float x[CH]; unsigned u[CH]; f2v p[CH];                                                              \
        _Pragma("unroll") for (int c = 0; c < CH; ++c) { x[c] = 1.f + threadIdx.x * 1e-3f + c; u[c] = threadIdx.x + c; p[c] = f2v{x[c], 1.f}; } \
        const float y = 1.0001f; const unsigned w = 0x7fffu; const f2v q = f2v{1.0001f, 0.9999f};        \
        for (int it = 0; it < n; ++it) {                                                                     \
            _Pragma("unroll") for (int r = 0; r < 16 / CH * 1 + (CH > 16 ? 1 : 0); ++r)                     \
            _Pragma("unroll") for (int c = 0; c < CH; ++c) { ASM; }                                         \
        }                                                                                                    \
        float s = 0.f;                                                                                       \
        _Pragma("unroll") for (int c = 0; c < CH; ++c) s += x[c] + (float)u[c] + p[c][0];                   \
        if (s == 1.2345f) out[blockIdx.x * 256 + threadIdx.x] = s;                                           \
    }
// every op is applied 16 times per iteration per chain set (CH = 1: one dependent chain of 16; CH = 8: 2 x 8)
VOP(fma, asm volatile("v_fma_f32 %0, %0, %1, 0.5" : "+v"(x[c]) : "v"(y)))
VOP(exp, asm volatile("v_exp_f32 %0, %0" : "+v"(x[c])))
VOP(rcp, asm volatile("v_rcp_f32 %0, %0" : "+v"(x[c])))
VOP(log, asm volatile("v_log_f32 %0, %0" : "+v"(x[c])))
VOP(pkfma, asm volatile("v_pk_fma_f32 %0, %0, %1, %1" : "+v"(p[c]) : "v"(q)))
VOP(pkmul, asm volatile("v_pk_mul_f32 %0, %0, %1" : "+v"(p[c]) : "v"(q)))
VOP(add3, asm volatile("v_add3_u32 %0, %0, %1, %0" : "+v"(u[c]) : "v"(w)))
VOP(bfe, asm volatile("v_bfe_u32 %0, %0, 16, 1" : "+v"(u[c])))
VOP(perm, asm volatile("v_perm_b32 %0, %0, %1, %1" : "+v"(u[c]) : "v"(w)))
VOP(dpp, asm volatile("v_mov_b32_dpp %0, %0 row_shr:1 bound_ctrl:0" : "+v"(u[c])))
VOP(cvtf16, asm volatile("v_cvt_f16_f32 %0, %0" : "+v"(x[c])))
// production bf16 RNE (g2_common.h bf_rne): v_bfe_u32 + v_add3_u32 (+ the shift / pack done by the consumer)
VOP(rne, { unsigned t; asm volatile("v_bfe_u32 %0, %1, 16, 1\n v_add3_u32 %1, %1, %2, %0" : "=&v"(t), "+v"(u[c]) : "v"(w)); })

template <class F>
static float best_ms(F launch, int reps = 5) {
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    launch(); hipDeviceSynchronize();
    float best = 1e30f;
    for (int r = 0; r < reps; ++r) {
        hipEventRecord(a); launch(); hipEventRecord(b); hipEventSynchronize(b);
        float ms; hipEventElapsedTime(&ms, a, b); best = ms < best ? ms : best;
    }
    return best;
}

template <int FORM, int I, int V>
static void mm(float* out) {
    const int n = 4000;
    for (int wps : {1, 2, 4}) {
        const float ms = best_ms([&] { hipLaunchKernelGGL((k_mmac<FORM, I, V>), dim3(80 * wps), dim3(256), 0, 0, n, out); });
        const double per_wave = ms * 1e-3 * clk / ((double)n * I);   // cycles between MMACs of one wave
        printf("MMAC form=%s indep=%d valu_per_mmac=%d waves_per_simd=%d cyc_per_mmac_wave=%.2f cyc_per_mmac_simd=%.2f\n",
               FORM ? "f32_16x16x8" : "bf16_16x16x16", I, V, wps, per_wave, per_wave / wps);
        fflush(stdout);
    }
}

#define RUNV(NAME)                                                                                           \
    {                                                                                                        \
        const int n = 20000;                                                                                 \
        const float lat = best_ms([&] { hipLaunchKernelGGL(k_##NAME<1>, dim3(80), dim3(64), 0, 0, n, out); }); \
        const float thr = best_ms([&] { hipLaunchKernelGGL(k_##NAME<8>, dim3(80 * 4), dim3(256), 0, 0, n, out); }); \
        printf("VALU op=%s lat_cyc=%.2f thr_cyc_per_inst_simd=%.3f\n", #NAME, lat * 1e-3 * clk / (n * 16.0),   \
               thr * 1e-3 * clk / (n * 16.0 * 4 /*waves per SIMD*/));                                        \
        fflush(stdout);                                                                                      \
    }

int main(int argc, char** argv) {
    if (argc > 1) clk = atof(argv[1]) * 1e9;
    float* out; hipMalloc(&out, 80 * 4 * 256 * 4);
    printf("# clk=%.3f GHz\n", clk / 1e9);
    mm<0, 1, 0>(out); mm<0, 2, 0>(out); mm<0, 4, 0>(out); mm<0, 8, 0>(out); mm<0, 16, 0>(out);
    mm<1, 1, 0>(out); mm<1, 2, 0>(out); mm<1, 4, 0>(out); mm<1, 8, 0>(out);
    mm<0, 4, 2>(out); mm<0, 4, 4>(out); mm<0, 4, 8>(out);
    RUNV(fma) RUNV(exp) RUNV(rcp) RUNV(log) RUNV(pkfma) RUNV(pkmul) RUNV(add3) RUNV(bfe) RUNV(perm) RUNV(dpp)
    RUNV(cvtf16) RUNV(rne)
    return 0;
}
