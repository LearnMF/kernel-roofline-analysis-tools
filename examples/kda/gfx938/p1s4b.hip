#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
typedef short v4bh __attribute__((ext_vector_type(4)));
typedef short v8bh __attribute__((ext_vector_type(8)));
typedef float f4 __attribute__((ext_vector_type(4)));
typedef int v4i __attribute__((ext_vector_type(4)));
constexpr int NF = 16;   // fragment pairs per wave (A, B) -> 16 KB per wave, 64 KB per CTA from HBM
__device__ __forceinline__ f4 mmac(v4bh a, v4bh b, f4 c) { return __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(a, b, c, false, false); }
// native fragment block: 256 bf16 = lane l's 4 elements at 4l (512 B); a wave's NF A blocks contiguous, likewise B
__device__ __forceinline__ size_t wbase(int cta, int w) { return ((size_t)cta * 4 + w) * NF * 256; }
// (a) per-lane 8 B fragment loads straight from HBM (the current s4b style)
__global__ void __launch_bounds__(256, 2) ka(const short* A, const short* B, float* out) {
  const int w = threadIdx.x >> 6, l = threadIdx.x & 63; const size_t b0 = wbase(blockIdx.x, w);
  v4bh fa[NF], fb[NF];
  #pragma unroll
  for (int f = 0; f < NF; ++f) { fa[f] = *reinterpret_cast<const v4bh*>(A + b0 + f * 256 + 4 * l); fb[f] = *reinterpret_cast<const v4bh*>(B + b0 + f * 256 + 4 * l); }
  f4 acc = {0, 0, 0, 0};
  #pragma unroll
  for (int f = 0; f < NF; ++f) acc = mmac(fa[f], fb[f], acc);
  out[(size_t)blockIdx.x * 256 + threadIdx.x] = acc[0] + acc[1] + acc[2] + acc[3];
}
// (c) per-lane 16 B loads of fragment PAIRS (layout: blocks f, f+1 interleaved per lane, 8 elements at 8l)
__global__ void __launch_bounds__(256, 2) kc(const short* A, const short* B, float* out) {
  const int w = threadIdx.x >> 6, l = threadIdx.x & 63; const size_t b0 = wbase(blockIdx.x, w);
  v8bh pa[NF / 2], pb[NF / 2];
  #pragma unroll
  for (int f = 0; f < NF / 2; ++f) { pa[f] = *reinterpret_cast<const v8bh*>(A + b0 + f * 512 + 8 * l); pb[f] = *reinterpret_cast<const v8bh*>(B + b0 + f * 512 + 8 * l); }
  f4 acc = {0, 0, 0, 0};
  #pragma unroll
  for (int f = 0; f < NF / 2; ++f) {
    acc = mmac(__builtin_shufflevector(pa[f], pa[f], 0, 1, 2, 3), __builtin_shufflevector(pb[f], pb[f], 0, 1, 2, 3), acc);
    acc = mmac(__builtin_shufflevector(pa[f], pa[f], 4, 5, 6, 7), __builtin_shufflevector(pb[f], pb[f], 4, 5, 6, 7), acc);
  }
  out[(size_t)blockIdx.x * 256 + threadIdx.x] = acc[0] + acc[1] + acc[2] + acc[3];
}
// (b) MLS bulk copy (matrix_load_32x32: 2 KB = 4 native blocks per instruction) into LDS, double-buffered groups of
// 4 pairs, then lane-contiguous ds_read_b64 fragments
__device__ __forceinline__ void mls2k(const short* g, short* lds) {
  uint64_t a = (uint64_t)g & 0xffffffffffffull;
  v4i r; r[0] = __builtin_amdgcn_readfirstlane((int)(uint32_t)a); r[1] = __builtin_amdgcn_readfirstlane((int)(uint32_t)(a >> 32)); r[2] = 32; r[3] = 0;
  const int so = __builtin_amdgcn_readfirstlane((int)(size_t)lds);
  __builtin_hcu_matrix_load_32x32_b16(r, (__attribute__((address_space(3))) short*)(so), 0, 0, 0, 0, 0);
}
__global__ void __launch_bounds__(256, 2) kb(const short* A, const short* B, float* out) {
  __shared__ __align__(16) short s[4][2][2][1024];   // wave x buffer x (A, B) x 4 blocks
  const int w = threadIdx.x >> 6, l = threadIdx.x & 63; const size_t b0 = wbase(blockIdx.x, w);
  f4 acc = {0, 0, 0, 0};
  mls2k(A + b0, s[w][0][0]); mls2k(B + b0, s[w][0][1]);
  #pragma unroll
  for (int gi = 0; gi < NF / 4; ++gi) {
    const int cb = gi & 1;
    if (gi + 1 < NF / 4) { mls2k(A + b0 + (gi + 1) * 1024, s[w][cb ^ 1][0]); mls2k(B + b0 + (gi + 1) * 1024, s[w][cb ^ 1][1]); }
    if (gi + 1 < NF / 4) asm volatile("s_waitcnt vmcnt(2)" ::: "memory"); else asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    v4bh fa[4], fb[4];
    #pragma unroll
    for (int f = 0; f < 4; ++f) { fa[f] = *reinterpret_cast<const v4bh*>(&s[w][cb][0][f * 256 + 4 * l]); fb[f] = *reinterpret_cast<const v4bh*>(&s[w][cb][1][f * 256 + 4 * l]); }
    #pragma unroll
    for (int f = 0; f < 4; ++f) acc = mmac(fa[f], fb[f], acc);
  }
  out[(size_t)blockIdx.x * 256 + threadIdx.x] = acc[0] + acc[1] + acc[2] + acc[3];
}
// (d) per-lane 4 B loads (two per fragment half) -- the narrow-load TA cost per byte
__global__ void __launch_bounds__(256, 2) kd(const short* A, const short* B, float* out) {
  const int w = threadIdx.x >> 6, l = threadIdx.x & 63; const size_t b0 = wbase(blockIdx.x, w);
  int fa[NF][2], fb[NF][2];
  #pragma unroll
  for (int f = 0; f < NF; ++f) { const int* pa = reinterpret_cast<const int*>(A + b0 + f * 256 + 4 * l); const int* pb = reinterpret_cast<const int*>(B + b0 + f * 256 + 4 * l);
    fa[f][0] = __builtin_nontemporal_load(pa); fa[f][1] = __builtin_nontemporal_load(pa + 1); fb[f][0] = __builtin_nontemporal_load(pb); fb[f][1] = __builtin_nontemporal_load(pb + 1); }
  f4 acc = {0, 0, 0, 0};
  #pragma unroll
  for (int f = 0; f < NF; ++f) { v4bh a, b; int* ia = reinterpret_cast<int*>(&a); int* ib = reinterpret_cast<int*>(&b); ia[0] = fa[f][0]; ia[1] = fa[f][1]; ib[0] = fb[f][0]; ib[1] = fb[f][1]; acc = mmac(a, b, acc); }
  out[(size_t)blockIdx.x * 256 + threadIdx.x] = acc[0] + acc[1] + acc[2] + acc[3];
}
int main(int argc, char** argv) {
  const int NC = argc > 1 ? atoi(argv[1]) : 8192;
  const size_t n = (size_t)NC * 4 * NF * 256;
  std::vector<short> h(n); for (size_t i = 0; i < n; ++i) h[i] = (short)(0x3f80 + (i % 7));
  short *dA, *dB; float *o1, *o2, *o3; hipMalloc(&dA, n * 2); hipMalloc(&dB, n * 2); hipMalloc(&o1, NC * 1024); hipMalloc(&o2, NC * 1024); hipMalloc(&o3, NC * 1024);
  hipMemcpy(dA, h.data(), n * 2, hipMemcpyHostToDevice); hipMemcpy(dB, h.data(), n * 2, hipMemcpyHostToDevice);
  hipLaunchKernelGGL(ka, dim3(NC), dim3(256), 0, 0, dA, dB, o1); hipLaunchKernelGGL(kb, dim3(NC), dim3(256), 0, 0, dA, dB, o2); hipDeviceSynchronize();
  std::vector<float> r1(NC * 256), r2(NC * 256); hipMemcpy(r1.data(), o1, NC * 1024, hipMemcpyDeviceToHost); hipMemcpy(r2.data(), o2, NC * 1024, hipMemcpyDeviceToHost);
  size_t diff = 0; for (size_t i = 0; i < r1.size(); ++i) diff += r1[i] != r2[i];
  printf("a vs b (same fragment values, same order): %zu differ\n", diff);
  hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
  const char* nm[4] = {"a per-lane 8B fragment loads ", "b MLS 2KB -> LDS -> ds_read_b64", "c per-lane 16B pair loads    ", "d per-lane 4B loads          "};
  for (int v = 0; v < 4; ++v) {
    float best = 1e9;
    for (int rep = 0; rep < 20; ++rep) {
      hipEventRecord(e0);
      if (v == 0) hipLaunchKernelGGL(ka, dim3(NC), dim3(256), 0, 0, dA, dB, o1); else if (v == 1) hipLaunchKernelGGL(kb, dim3(NC), dim3(256), 0, 0, dA, dB, o2); else if (v == 2) hipLaunchKernelGGL(kc, dim3(NC), dim3(256), 0, 0, dA, dB, o3); else hipLaunchKernelGGL(kd, dim3(NC), dim3(256), 0, 0, dA, dB, o3);
      hipEventRecord(e1); hipEventSynchronize(e1); float ms; hipEventElapsedTime(&ms, e0, e1); best = fmin(best, ms);
    }
    printf("%s: %7.1f us  %6.0f GB/s\n", nm[v], best * 1e3, 2.0 * n * 2 / (best * 1e-3) / 1e9);
  }
}
