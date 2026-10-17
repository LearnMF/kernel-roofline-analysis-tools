// Two semantics questions about gfx936 LDS-DMA (buffer_load_dwordx4 ... lds), one wave per block:
//  T1 (M0 timing): with the VMEM queue busy (8 outstanding global loads), DMA#1 with M0 = X, then s_mov m0 = Y and
//     DMA#2.  Does DMA#1 still land at X (M0 latched at issue) or at Y (read late)?
//  T2 (offset): one DMA with offset:1024 and M0 = X -- where does it land, and which global bytes does it read?
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
typedef int v4i __attribute__((ext_vector_type(4)));
__device__ __forceinline__ v4i mk_rsrc(const void* p, unsigned bytes) {
  const unsigned long a = (unsigned long)p; v4i r;
  r.x = __builtin_amdgcn_readfirstlane((int)(a & 0xffffffffu)); r.y = __builtin_amdgcn_readfirstlane((int)(a >> 32));
  r.z = __builtin_amdgcn_readfirstlane((int)bytes); r.w = 0x00020000; return r;
}
__global__ void t1(const float* a, const float* b, const float* junk, float* out, int busy) {
  __shared__ __attribute__((aligned(16))) float lds[3 * 256];   // X at 0, Y at 1 KB, spare
  for (int i = threadIdx.x; i < 3 * 256; i += 64) lds[i] = -1.f;
  __syncthreads();
  const v4i ra = mk_rsrc(a, 1024), rb = mk_rsrc(b, 1024);
  const int X = __builtin_amdgcn_readfirstlane((int)(size_t)&lds[0]), Y = __builtin_amdgcn_readfirstlane((int)(size_t)&lds[256]);
  const int vo = threadIdx.x * 16, z = 0;
  float4 j[8];
  if (busy) {
#pragma unroll
    for (int u = 0; u < 8; ++u) j[u] = reinterpret_cast<const float4*>(junk)[u * 64 * 1024 + threadIdx.x * 1024];
  }
  asm volatile("s_mov_b32 m0, %0\n\ts_nop 0\n\tbuffer_load_dwordx4 %1, %2, %3 offen lds\n\t"
               "s_mov_b32 m0, %4\n\ts_nop 0\n\tbuffer_load_dwordx4 %1, %5, %3 offen lds\n\ts_waitcnt vmcnt(0)"
               :: "s"(X), "v"(vo), "s"(ra), "s"(z), "s"(Y), "s"(rb) : "memory", "m0");
  __syncthreads();
  float s = 0.f;
  if (busy) for (int u = 0; u < 8; ++u) s += j[u].x;
  for (int i = threadIdx.x; i < 3 * 256; i += 64) out[i] = lds[i];
  if (s == 12345.f) out[0] = s;
}
__global__ void t2(const float* a, float* out) {
  __shared__ __attribute__((aligned(16))) float lds[3 * 256];
  for (int i = threadIdx.x; i < 3 * 256; i += 64) lds[i] = -1.f;
  __syncthreads();
  const v4i ra = mk_rsrc(a, 4096);
  const int X = __builtin_amdgcn_readfirstlane((int)(size_t)&lds[0]);
  const int vo = threadIdx.x * 16, z = 0;
  asm volatile("s_mov_b32 m0, %0\n\ts_nop 0\n\tbuffer_load_dwordx4 %1, %2, %3 offen offset:1024 lds\n\ts_waitcnt vmcnt(0)"
               :: "s"(X), "v"(vo), "s"(ra), "s"(z) : "memory", "m0");
  __syncthreads();
  for (int i = threadIdx.x; i < 3 * 256; i += 64) out[i] = lds[i];
}
int main() {
  std::vector<float> ha(1024), hb(1024);
  for (int i = 0; i < 1024; ++i) { ha[i] = 1000.f + i; hb[i] = 5000.f + i; }
  float *a, *b, *junk, *o; hipMalloc(&a, 4096 * 4); hipMalloc(&b, 4096); hipMalloc(&junk, 512L << 20); hipMalloc(&o, 3 * 256 * 4);
  hipMemcpy(a, ha.data(), 4096, hipMemcpyHostToDevice); hipMemcpy(b, hb.data(), 4096, hipMemcpyHostToDevice);
  std::vector<float> big(1024); for (int i = 0; i < 1024; ++i) big[i] = 1000.f + i;
  hipMemcpy(a, big.data(), 4096, hipMemcpyHostToDevice);
  std::vector<float> r(768);
  for (int busy : {0, 1}) {
    int badX = 0, badY = 0;
    for (int rep = 0; rep < 200; ++rep) {
      hipLaunchKernelGGL(t1, dim3(1), dim3(64), 0, 0, a, b, junk, o, busy); hipMemcpy(r.data(), o, 768 * 4, hipMemcpyDeviceToHost);
      for (int i = 0; i < 256; ++i) { badX += r[i] != 1000.f + i; badY += r[256 + i] != 5000.f + i; }
    }
    printf("T1 busy=%d: DMA#1 (M0=X) wrong elems %d, DMA#2 (M0=Y) wrong elems %d over 200 runs; X[0]=%g Y[0]=%g\n", busy, badX, badY, r[0], r[256]);
  }
  std::vector<float> big2(1024); for (int i = 0; i < 1024; ++i) big2[i] = 1000.f + i;
  float* a2; hipMalloc(&a2, 8192); std::vector<float> h2(2048); for (int i = 0; i < 2048; ++i) h2[i] = 1000.f + i;
  hipMemcpy(a2, h2.data(), 8192, hipMemcpyHostToDevice);
  hipLaunchKernelGGL(t2, dim3(1), dim3(64), 0, 0, a2, o); hipMemcpy(r.data(), o, 768 * 4, hipMemcpyDeviceToHost);
  int first = -1; for (int i = 0; i < 768; ++i) if (r[i] != -1.f) { first = i; break; }
  printf("T2 offset:1024: first written LDS float index %d (0 = offset ignored for LDS, 256 = applied), value there %g "
         "(1000 = global offset ignored, 1256 = applied)\n", first, first >= 0 ? r[first] : -1.f);
}
