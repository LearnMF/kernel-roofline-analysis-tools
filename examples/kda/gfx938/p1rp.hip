// s4b-like regime: ~220 live VGPRs (2 waves / SIMD), phases separated by barriers, a few MMACs + gating VALU per
// phase, operands streamed from HBM each phase.  A: per-lane loads at phase start (what the register budget allows).
// B: MLS prefetch of the NEXT phase into LDS (double buffer, no VGPRs), fragments via ds_read_b64.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
typedef short v4bh __attribute__((ext_vector_type(4)));
typedef float f4 __attribute__((ext_vector_type(4)));
typedef int v4i __attribute__((ext_vector_type(4)));
#ifndef ACC
#define ACC 8
#endif
#ifndef NPR
#define NPR 160           // pinned live VGPRs (register pressure)
#endif
constexpr int P = 8;      // phases per CTA
constexpr int FP = 4;     // fragment pairs per wave per phase (4 KB per wave per phase)
__device__ __forceinline__ f4 mmac(v4bh a, v4bh b, f4 c) { return __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(a, b, c, false, false); }
__device__ __forceinline__ short bf_rne(float x) { uint32_t u = __float_as_uint(x); u += 0x7fffu + ((u >> 16) & 1u); return (short)(u >> 16); }
__device__ __forceinline__ float bf2f(short s) { return __uint_as_float(((uint32_t)(unsigned short)s) << 16); }
__device__ __forceinline__ size_t base(int cta, int w, int p) { return (((size_t)cta * 4 + w) * P + p) * FP * 256; }
__device__ __forceinline__ v4bh gate(v4bh x, float s) {   // a little gating VALU per fragment (exp2 + mul + rne)
  v4bh y; float e = __builtin_amdgcn_exp2f(s);
  #pragma unroll
  for (int i = 0; i < 4; ++i) y[i] = bf_rne(bf2f(x[i]) * e);
  return y;
}
__global__ void __launch_bounds__(256) kA(const short* A, const short* B, const float* G, float* out) {
  const int w = threadIdx.x >> 6, l = threadIdx.x & 63;
  f4 acc[ACC];
  float pr[NPR];
  #pragma unroll
  for (int i = 0; i < NPR; ++i) pr[i] = (float)(threadIdx.x + i);
  #pragma unroll
  for (int i = 0; i < ACC; ++i) acc[i] = f4{0, 0, 0, (float)i};
  #pragma unroll
  for (int p = 0; p < P; ++p) {
    const size_t b0 = base(blockIdx.x, w, p);
    v4bh fa[FP], fb[FP];
    #pragma unroll
    for (int f = 0; f < FP; ++f) { fa[f] = *reinterpret_cast<const v4bh*>(A + b0 + f * 256 + 4 * l); fb[f] = *reinterpret_cast<const v4bh*>(B + b0 + f * 256 + 4 * l); }
    const float s = G[((size_t)blockIdx.x * P + p) * 256 + threadIdx.x];
    #pragma unroll
    for (int f = 0; f < FP; ++f) { const int j = (p * FP + f) % ACC; acc[j] = mmac(gate(fa[f], s), fb[f], acc[j]); }
    #pragma unroll
    for (int i = 0; i < NPR; ++i) asm volatile("" : "+v"(pr[i]));
    __syncthreads();
  }
  float r = 0;
  #pragma unroll
  for (int i = 0; i < ACC; ++i) r += acc[i][0] + acc[i][1] + acc[i][2] + acc[i][3];
  #pragma unroll
  for (int i = 0; i < NPR; ++i) r += pr[i];
  out[(size_t)blockIdx.x * 256 + threadIdx.x] = r;
}
__device__ __forceinline__ void mls2k(const short* g, short* lds) {
  uint64_t a = (uint64_t)g & 0xffffffffffffull;
  v4i r; r[0] = __builtin_amdgcn_readfirstlane((int)(uint32_t)a); r[1] = __builtin_amdgcn_readfirstlane((int)(uint32_t)(a >> 32)); r[2] = 32; r[3] = 0;
  const int so = __builtin_amdgcn_readfirstlane((int)(size_t)lds);
  __builtin_hcu_matrix_load_32x32_b16(r, (__attribute__((address_space(3))) short*)(so), 0, 0, 0, 0, 0);
}
__global__ void __launch_bounds__(256) kB(const short* A, const short* B, const float* G, float* out) {
  __shared__ __align__(16) short s[4][2][2][1024];   // wave x buffer x (A, B) x FP blocks
  const int w = threadIdx.x >> 6, l = threadIdx.x & 63;
  f4 acc[ACC];
  float pr[NPR];
  #pragma unroll
  for (int i = 0; i < NPR; ++i) pr[i] = (float)(threadIdx.x + i);
  #pragma unroll
  for (int i = 0; i < ACC; ++i) acc[i] = f4{0, 0, 0, (float)i};
  mls2k(A + base(blockIdx.x, w, 0), s[w][0][0]); mls2k(B + base(blockIdx.x, w, 0), s[w][0][1]);
  #pragma unroll
  for (int p = 0; p < P; ++p) {
    const int cb = p & 1;
    const float sg = G[((size_t)blockIdx.x * P + p) * 256 + threadIdx.x];
    if (p + 1 < P) { mls2k(A + base(blockIdx.x, w, p + 1), s[w][cb ^ 1][0]); mls2k(B + base(blockIdx.x, w, p + 1), s[w][cb ^ 1][1]); }
    if (p + 1 < P) asm volatile("s_waitcnt vmcnt(2)" ::: "memory"); else asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    v4bh fa[FP], fb[FP];
    #pragma unroll
    for (int f = 0; f < FP; ++f) { fa[f] = *reinterpret_cast<const v4bh*>(&s[w][cb][0][f * 256 + 4 * l]); fb[f] = *reinterpret_cast<const v4bh*>(&s[w][cb][1][f * 256 + 4 * l]); }
    #pragma unroll
    for (int f = 0; f < FP; ++f) { const int j = (p * FP + f) % ACC; acc[j] = mmac(gate(fa[f], sg), fb[f], acc[j]); }
    #pragma unroll
    for (int i = 0; i < NPR; ++i) asm volatile("" : "+v"(pr[i]));
    __syncthreads();
  }
  float r = 0;
  #pragma unroll
  for (int i = 0; i < ACC; ++i) r += acc[i][0] + acc[i][1] + acc[i][2] + acc[i][3];
  #pragma unroll
  for (int i = 0; i < NPR; ++i) r += pr[i];
  out[(size_t)blockIdx.x * 256 + threadIdx.x] = r;
}
int main(int argc, char** argv) {
  const int NC = argc > 1 ? atoi(argv[1]) : 4096;
  const size_t n = (size_t)NC * 4 * P * FP * 256;
  std::vector<short> h(n); for (size_t i = 0; i < n; ++i) h[i] = (short)(0x3f80 + (i % 11));
  std::vector<float> hg((size_t)NC * P * 256); for (size_t i = 0; i < hg.size(); ++i) hg[i] = (i % 13) * 0.01f;
  short *dA, *dB; float *dG, *o1, *o2;
  hipMalloc(&dA, n * 2); hipMalloc(&dB, n * 2); hipMalloc(&dG, hg.size() * 4); hipMalloc(&o1, NC * 1024); hipMalloc(&o2, NC * 1024);
  hipMemcpy(dA, h.data(), n * 2, hipMemcpyHostToDevice); hipMemcpy(dB, h.data(), n * 2, hipMemcpyHostToDevice); hipMemcpy(dG, hg.data(), hg.size() * 4, hipMemcpyHostToDevice);
  hipLaunchKernelGGL(kA, dim3(NC), dim3(256), 0, 0, dA, dB, dG, o1); hipLaunchKernelGGL(kB, dim3(NC), dim3(256), 0, 0, dA, dB, dG, o2); hipDeviceSynchronize();
  std::vector<float> r1(NC * 256), r2(NC * 256); hipMemcpy(r1.data(), o1, NC * 1024, hipMemcpyDeviceToHost); hipMemcpy(r2.data(), o2, NC * 1024, hipMemcpyDeviceToHost);
  size_t diff = 0; for (size_t i = 0; i < r1.size(); ++i) diff += r1[i] != r2[i];
  hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
  float t[2];
  for (int v = 0; v < 2; ++v) {
    float best = 1e9;
    for (int rep = 0; rep < 20; ++rep) {
      hipEventRecord(e0);
      if (v == 0) hipLaunchKernelGGL(kA, dim3(NC), dim3(256), 0, 0, dA, dB, dG, o1); else hipLaunchKernelGGL(kB, dim3(NC), dim3(256), 0, 0, dA, dB, dG, o2);
      hipEventRecord(e1); hipEventSynchronize(e1); float ms; hipEventElapsedTime(&ms, e0, e1); best = fmin(best, ms);
    }
    t[v] = best;
  }
  printf("NPR=%d: A per-lane %.1f us (%.0f GB/s) | B MLS prefetch %.1f us (%.0f GB/s) | B/A speedup %.2fx | bitwise diff %zu\n",
         NPR, t[0] * 1e3, 2.0 * n * 2 / (t[0] * 1e-3) / 1e9, t[1] * 1e3, 2.0 * n * 2 / (t[1] * 1e-3) / 1e9, t[0] / t[1], diff);
}
