// In-kernel MLS vs per-lane loads, s4a pass-W shaped: 8 steps x 4 KB block, barrier per step, 4 dwT-like 2 B stores per
// lane per step, 8 x ld8 LDS reads per lane + 8 MMAC per step.  Variants: A per-lane 16 B (reg prefetch 1 step ahead),
// B MLS 2 KB tiles from 2 waves (3-step lead ring), C MLS 1 KB tiles (32x16) from 4 waves, D = B without stores.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
typedef short v4s __attribute__((ext_vector_type(4)));
typedef float f4 __attribute__((ext_vector_type(4)));
typedef int v4i __attribute__((ext_vector_type(4)));
__device__ __forceinline__ void mls(const void* g, int rows32, void* lds) {   // rows32: 1 -> 32x32 (2 KB), 0 -> 32x16 (1 KB)
  const uint64_t a = reinterpret_cast<uint64_t>(g) & 0xffffffffffffull; v4i r;
  r[0] = __builtin_amdgcn_readfirstlane((int)(uint32_t)a); r[1] = __builtin_amdgcn_readfirstlane((int)(uint32_t)(a >> 32));
  r[2] = rows32 ? 32 : 16; r[3] = 0; const int so = __builtin_amdgcn_readfirstlane((int)(size_t)lds);
  if (rows32) __builtin_hcu_matrix_load_32x32_b16(r, (__attribute__((address_space(3))) short*)(so), 0, 0, 0, 0, 0);
  else __builtin_hcu_matrix_load_32x16_b16(r, (__attribute__((address_space(3))) short*)(so), 0, 0, 0, 0, 0);
}
template <int N> __device__ __forceinline__ void waitvm() {
  if constexpr (N >= 15) asm volatile("s_waitcnt vmcnt(15)" ::: "memory");
  else if constexpr (N == 14) asm volatile("s_waitcnt vmcnt(14)" ::: "memory");
  else if constexpr (N == 13) asm volatile("s_waitcnt vmcnt(13)" ::: "memory");
  else if constexpr (N == 12) asm volatile("s_waitcnt vmcnt(12)" ::: "memory");
  else if constexpr (N == 11) asm volatile("s_waitcnt vmcnt(11)" ::: "memory");
  else if constexpr (N == 10) asm volatile("s_waitcnt vmcnt(10)" ::: "memory");
  else if constexpr (N == 9) asm volatile("s_waitcnt vmcnt(9)" ::: "memory");
  else if constexpr (N == 8) asm volatile("s_waitcnt vmcnt(8)" ::: "memory");
  else if constexpr (N == 7) asm volatile("s_waitcnt vmcnt(7)" ::: "memory");
  else if constexpr (N == 6) asm volatile("s_waitcnt vmcnt(6)" ::: "memory");
  else if constexpr (N == 5) asm volatile("s_waitcnt vmcnt(5)" ::: "memory");
  else asm volatile("s_waitcnt vmcnt(4)" ::: "memory");
}
template <int KT> __device__ __forceinline__ void wait_exact() {   // younger than MLS(KT+1): its later MLS + later stores
  constexpr int it0 = KT - 2 > 0 ? KT - 2 : 0, iters = KT - it0 + 1;
  constexpr int ml = (KT + 3 < 7 ? KT + 3 : 7) - (KT + 1);
  waitvm<4 * iters + ml>();
}
template <int V>
__global__ void __launch_bounds__(256) k(const short* __restrict__ h, short* __restrict__ out, float* acc_out) {
  __shared__ __align__(16) short ring[4][2048];
  const int tid = threadIdx.x, w = tid >> 6, l = tid & 63;
  const short* hb = h + (size_t)blockIdx.x * 8 * 2048;
  short* ob = out + (size_t)blockIdx.x * 8 * 1024;
  f4 acc = {0, 0, 0, 0}; v4s af = {1, 1, 1, 1};
  f4 pre;
  if (V == 0 || V == 5) { *reinterpret_cast<f4*>(&ring[0][8 * tid]) = *reinterpret_cast<const f4*>(hb + 8 * tid); pre = *reinterpret_cast<const f4*>(hb + 2048 + 8 * tid); }
  else if (V == 2) { for (int s = 0; s < 4; ++s) mls(hb + s * 2048 + w * 512, 0, &ring[s][w * 512]); asm volatile("s_waitcnt vmcnt(3)" ::: "memory"); }
  else { if (w < 2) { for (int s = 0; s < 4; ++s) mls(hb + s * 2048 + w * 1024, 1, &ring[s][w * 1024]); asm volatile("s_waitcnt vmcnt(3)" ::: "memory"); } }
  __syncthreads();
  #pragma unroll
  for (int kt = 0; kt < 8; ++kt) {
    const int kk = kt;
    const short* hk = &ring[kt & 3][l * 4];
    #pragma unroll
    for (int vt = 0; vt < 8; ++vt) acc = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(af, *reinterpret_cast<const v4s*>(hk + vt * 256), acc, 0, 0);
    if ((V == 0 || V == 5) && kt < 7) { *reinterpret_cast<f4*>(&ring[(kt + 1) & 3][8 * tid]) = pre; if (kt < 6) pre = *reinterpret_cast<const f4*>(hb + (kt + 2) * 2048 + 8 * tid); }
    if (V != 3 && V != 5) for (int e = 0; e < 4; ++e) ob[kt * 1024 + (4 * (l >> 4) + e) * 64 + 16 * w + (l & 15)] = (short)(acc[e]);
    if (V != 0 && V != 5 && kt < 7 && (V == 2 || w < 2)) {
      if (V == 3) asm volatile("s_waitcnt vmcnt(2)" ::: "memory");
      else if (V == 4) { switch (kk) { case 0: wait_exact<0>(); break; case 1: wait_exact<1>(); break; case 2: wait_exact<2>(); break;
                                       case 3: wait_exact<3>(); break; case 4: wait_exact<4>(); break; case 5: wait_exact<5>(); break; default: wait_exact<6>(); } }
      else asm volatile("s_waitcnt vmcnt(6)" ::: "memory");
    }
    __syncthreads();
    if (V != 0 && V != 5 && kt + 4 < 8) {
      if (V == 2) mls(hb + (kt + 4) * 2048 + w * 512, 0, &ring[kt & 3][w * 512]);
      else if (w < 2) mls(hb + (kt + 4) * 2048 + w * 1024, 1, &ring[kt & 3][w * 1024]);
    }
  }
  acc_out[blockIdx.x * 256 + tid] = acc[0] + acc[1] + acc[2] + acc[3];
}
int main() {
  const int NC = 8192; short *h, *o; float* a;
  hipMalloc(&h, (size_t)NC * 8 * 2048 * 2); hipMalloc(&o, (size_t)NC * 8 * 1024 * 2); hipMalloc(&a, NC * 256 * 4);
  hipMemset(h, 0, (size_t)NC * 8 * 2048 * 2);
  void (*ks[6])(const short*, short*, float*) = {k<0>, k<1>, k<2>, k<3>, k<4>, k<5>};
  const char* nm[6] = {"A per-lane 16B + reg prefetch", "B MLS 2KB, 2 waves", "C MLS 1KB, 4 waves", "D = B, no stores", "E = B, exact vmcnt (lead kept)", "F = A, no stores"};
  hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
  for (int rep = 0; rep < 2; ++rep) for (int v = 0; v < 6; ++v) {
    hipLaunchKernelGGL(ks[v], dim3(NC), dim3(256), 0, 0, h, o, a); hipDeviceSynchronize();
    hipEventRecord(e0); for (int i = 0; i < 10; ++i) hipLaunchKernelGGL(ks[v], dim3(NC), dim3(256), 0, 0, h, o, a); hipEventRecord(e1); hipEventSynchronize(e1);
    float ms; hipEventElapsedTime(&ms, e0, e1); printf("rep %d %-32s %8.1f us/launch\n", rep, nm[v], ms * 100);
  }
}
