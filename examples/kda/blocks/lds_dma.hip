// Does gfx936 execute direct-to-LDS buffer loads (buffer_load_dword{,x4} ... lds: global -> LDS without VGPR staging,
// the AMD counterpart of cp.async)?  Correctness + streaming bandwidth vs the register path.
//   hipcc -O3 --offload-arch=gfx936 lds_dma.hip -o lds_dma
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

typedef int v4i __attribute__((ext_vector_type(4)));
__device__ __forceinline__ v4i mk_rsrc(const void* p, unsigned bytes) {   // gfx9 raw buffer V# (CK's 0x00020000 dword3)
  const unsigned long a = (unsigned long)p;
  v4i r;
  r.x = __builtin_amdgcn_readfirstlane((int)(a & 0xffffffffu));
  r.y = __builtin_amdgcn_readfirstlane((int)(a >> 32));
  r.z = __builtin_amdgcn_readfirstlane((int)bytes);
  r.w = 0x00020000;
  return r;
}

template <int SZ>   // SZ = bytes per lane: 4 or 16
__global__ void k_dma(const float* __restrict__ in, float* out, int chunks) {
  __shared__ __attribute__((aligned(16))) float lds[256 * 4];
  float s = 0.f;
  for (int c = 0; c < chunks; ++c) {
    const char* base = (const char*)in + ((size_t)(blockIdx.x * chunks + c) * 256 * SZ);
    const v4i r = mk_rsrc(base, 256 * SZ);
    // M0 = this wave's LDS base; lane l's data lands at M0 + l*SZ
    const int ldsa = __builtin_amdgcn_readfirstlane((int)(size_t)((char*)lds + (threadIdx.x / 64) * 64 * SZ));
    const int voff = (int)threadIdx.x * SZ;
    if (SZ == 4)
      asm volatile("s_mov_b32 m0, %0\n\ts_nop 0\n\tbuffer_load_dword %1, %2, 0 offen lds\n\ts_waitcnt vmcnt(0)"
                   :: "s"(ldsa), "v"(voff), "s"(r) : "memory", "m0");
    else
      asm volatile("s_mov_b32 m0, %0\n\ts_nop 0\n\tbuffer_load_dwordx4 %1, %2, 0 offen lds\n\ts_waitcnt vmcnt(0)"
                   :: "s"(ldsa), "v"(voff), "s"(r) : "memory", "m0");
    __syncthreads();
    const float* l = lds;
    for (int e = 0; e < SZ / 4; ++e) s += l[threadIdx.x * (SZ / 4) + e];
    __syncthreads();
  }
  out[blockIdx.x * 256 + threadIdx.x] = s;
}

template <int SZ>
__global__ void k_reg(const float* __restrict__ in, float* out, int chunks) {
  float s = 0.f;
  for (int c = 0; c < chunks; ++c) {
    const float* base = in + (size_t)(blockIdx.x * chunks + c) * 256 * (SZ / 4);
    for (int e = 0; e < SZ / 4; ++e) s += base[threadIdx.x * (SZ / 4) + e];
  }
  out[blockIdx.x * 256 + threadIdx.x] = s;
}

int main() {
  const int blocks = 80 * 8, chunks = 256;
  for (int sz : {4, 16}) {
    const size_t n = (size_t)blocks * chunks * 256 * (sz / 4);
    std::vector<float> h(n);
    for (size_t i = 0; i < n; ++i) h[i] = (float)((i * 2654435761u) % 1000) * 0.001f;
    float *in, *o1, *o2;
    hipMalloc(&in, n * 4); hipMalloc(&o1, blocks * 256 * 4); hipMalloc(&o2, blocks * 256 * 4);
    hipMemcpy(in, h.data(), n * 4, hipMemcpyHostToDevice);
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    float tdma = 1e9, treg = 1e9;
    for (int r = 0; r < 6; ++r) {
      float ms;
      hipEventRecord(a);
      if (sz == 4) hipLaunchKernelGGL(k_dma<4>, dim3(blocks), dim3(256), 0, 0, in, o1, chunks);
      else hipLaunchKernelGGL(k_dma<16>, dim3(blocks), dim3(256), 0, 0, in, o1, chunks);
      hipEventRecord(b); hipEventSynchronize(b); hipEventElapsedTime(&ms, a, b); if (r) tdma = std::min(tdma, ms);
      hipEventRecord(a);
      if (sz == 4) hipLaunchKernelGGL(k_reg<4>, dim3(blocks), dim3(256), 0, 0, in, o2, chunks);
      else hipLaunchKernelGGL(k_reg<16>, dim3(blocks), dim3(256), 0, 0, in, o2, chunks);
      hipEventRecord(b); hipEventSynchronize(b); hipEventElapsedTime(&ms, a, b); if (r) treg = std::min(treg, ms);
    }
    std::vector<float> r1(blocks * 256), r2(blocks * 256);
    hipMemcpy(r1.data(), o1, r1.size() * 4, hipMemcpyDeviceToHost); hipMemcpy(r2.data(), o2, r2.size() * 4, hipMemcpyDeviceToHost);
    int bad = 0; for (size_t i = 0; i < r1.size(); ++i) bad += (r1[i] != r2[i]);
    printf("SZ=%2d B/lane: direct-to-LDS %s (mismatch %d/%zu); %.1f GB/s vs register path %.1f GB/s\n", sz,
           bad ? "WRONG" : "correct", bad, r1.size(), n * 4 / (tdma * 1e-3) / 1e9, n * 4 / (treg * 1e-3) / 1e9);
    hipFree(in); hipFree(o1); hipFree(o2);
  }
}
