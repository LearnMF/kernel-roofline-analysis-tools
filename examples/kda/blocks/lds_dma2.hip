// Pipelined direct-to-LDS (buffer_load_dwordx4 ... lds) vs register-staged loads at LOW occupancy (B blocks/CU),
// D slots in flight per block (each slot = 256 lanes x 16 B = 4 KB).  Same bytes, same consumption.
#include <hip/hip_runtime.h>
#include <cstdio>
typedef int v4i __attribute__((ext_vector_type(4)));
__device__ __forceinline__ v4i mk_rsrc(const void* p, unsigned bytes) {
  const unsigned long a = (unsigned long)p; v4i r;
  r.x = __builtin_amdgcn_readfirstlane((int)(a & 0xffffffffu)); r.y = __builtin_amdgcn_readfirstlane((int)(a >> 32));
  r.z = __builtin_amdgcn_readfirstlane((int)bytes); r.w = 0x00020000; return r;
}
template <int D>
__global__ void k_dma(const float4* __restrict__ in, float* out, int groups) {
  __shared__ __attribute__((aligned(16))) float4 lds[D][256];
  float s = 0.f;
  const v4i r = mk_rsrc(in, 0x7fffffff);
  for (int g = 0; g < groups; ++g) {
    const size_t base = ((size_t)(blockIdx.x * groups + g) * D) * 256;
#pragma unroll
    for (int d = 0; d < D; ++d) {
      const int ldsa = __builtin_amdgcn_readfirstlane((int)(size_t)(&lds[d][(threadIdx.x / 64) * 64]));
      const int soff = __builtin_amdgcn_readfirstlane((int)((base + (size_t)d * 256) * 16 / 1));  // < 2 GB
      const int voff = (int)(threadIdx.x * 16);
      asm volatile("s_mov_b32 m0, %0\n\ts_nop 0\n\tbuffer_load_dwordx4 %1, %2, %3 offen lds"
                   :: "s"(ldsa), "v"(voff), "s"(r), "s"(soff) : "memory", "m0");
    }
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __syncthreads();
#pragma unroll
    for (int d = 0; d < D; ++d) { const float4 v = lds[d][threadIdx.x]; s += v.x + v.y + v.z + v.w; }
    __syncthreads();
  }
  if (s == 1234.5f) out[blockIdx.x * 256 + threadIdx.x] = s;
}
template <int D>
__global__ void k_reg(const float4* __restrict__ in, float* out, int groups) {
  __shared__ __attribute__((aligned(16))) float4 lds[D][256];
  float s = 0.f;
  for (int g = 0; g < groups; ++g) {
    const size_t base = ((size_t)(blockIdx.x * groups + g) * D) * 256;
    float4 v[D];
#pragma unroll
    for (int d = 0; d < D; ++d) v[d] = in[base + (size_t)d * 256 + threadIdx.x];
#pragma unroll
    for (int d = 0; d < D; ++d) lds[d][threadIdx.x] = v[d];   // same staging through LDS as the DMA path
    __syncthreads();
#pragma unroll
    for (int d = 0; d < D; ++d) { const float4 w = lds[d][threadIdx.x]; s += w.x + w.y + w.z + w.w; }
    __syncthreads();
  }
  if (s == 1234.5f) out[blockIdx.x * 256 + threadIdx.x] = s;
}
template <int D> void run(const float4* in, float* out, int bpc, size_t n4) {
  const int blocks = 80 * bpc; const int groups = (int)(n4 / ((size_t)blocks * D * 256));
  hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b); float t1 = 1e9, t2 = 1e9, ms;
  for (int r = 0; r < 5; ++r) {
    hipEventRecord(a); hipLaunchKernelGGL(k_dma<D>, dim3(blocks), dim3(256), 0, 0, in, out, groups); hipEventRecord(b); hipEventSynchronize(b);
    hipEventElapsedTime(&ms, a, b); if (r) t1 = std::min(t1, ms);
    hipEventRecord(a); hipLaunchKernelGGL(k_reg<D>, dim3(blocks), dim3(256), 0, 0, in, out, groups); hipEventRecord(b); hipEventSynchronize(b);
    hipEventElapsedTime(&ms, a, b); if (r) t2 = std::min(t2, ms);
  }
  const double bytes = (double)groups * blocks * D * 256 * 16;
  printf("blocks/CU %d, %d x 4 KB in flight per block: direct-to-LDS %7.1f GB/s | register-staged %7.1f GB/s\n", bpc, D,
         bytes / (t1 * 1e-3) / 1e9, bytes / (t2 * 1e-3) / 1e9);
}
int main() {
  const size_t n4 = (size_t)1 << 27;   // 2 GB
  float4* in; float* out; hipMalloc(&in, n4 * 16); hipMalloc(&out, 80 * 8 * 256 * 4); hipMemset(in, 0, n4 * 16);
  for (int bpc : {1, 2}) { run<2>(in, out, bpc, n4); run<4>(in, out, bpc, n4); run<8>(in, out, bpc, n4); }
}
