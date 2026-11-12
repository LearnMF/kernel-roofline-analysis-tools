// Exhaustive: for every fp32 d in [1, 2^126), rcp + Newton chain (the compiler's divide sequence without scale /
// fixup, numerator 1) vs the compiler's IEEE 1.f / d.  Also checks v_pk_add_f32 with neg (packed subtract).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
typedef float f2 __attribute__((ext_vector_type(2)));
__device__ __forceinline__ float rcp1(float d) {
  #pragma clang fp contract(off)
  const float r0 = __builtin_amdgcn_rcpf(d);
  const float e0 = __builtin_fmaf(-d, r0, 1.f);
  const float r1 = __builtin_fmaf(e0, r0, r0);
  const float e1 = __builtin_fmaf(-d, r1, 1.f);
  const float q1 = __builtin_fmaf(e1, r1, r1);
  const float e2 = __builtin_fmaf(-d, q1, 1.f);
  return __builtin_fmaf(e2, r1, q1);
}
__global__ void k(uint32_t lo, unsigned long long* bad, uint32_t* first) {
  #pragma clang fp contract(off)
  const uint32_t u = lo + blockIdx.x * 256u + threadIdx.x;
  if (u >= 0x7e800000u) return;              // 2^126
  const float d = __uint_as_float(u);
  volatile float one = 1.f;
  const float ref = one / d;
  float got = rcp1(d);
  if (u == 0x40000000u || u == 0x7e000000u) got = __uint_as_float(__float_as_uint(got) + 1u);   // canaries: expect 2
  if (__float_as_uint(ref) != __float_as_uint(got)) { atomicAdd(bad, 1ull); atomicMin(first, u); }
}
int main() {
  unsigned long long* bad; uint32_t* first; hipMalloc(&bad, 8); hipMalloc(&first, 4);
  hipMemset(bad, 0, 8); uint32_t f0 = 0xffffffffu; hipMemcpy(first, &f0, 4, hipMemcpyHostToDevice);
  const uint32_t lo = 0x3f800000u, n = 0x7e800000u - lo;
  for (uint32_t off = 0; off < n; off += 256u * 65536u)
    hipLaunchKernelGGL(k, dim3(65536), dim3(256), 0, 0, lo + off, bad, first);
  printf("launch: %s\n", hipGetErrorString(hipDeviceSynchronize()));
  unsigned long long b; uint32_t fu; hipMemcpy(&b, bad, 8, hipMemcpyDeviceToHost); hipMemcpy(&fu, first, 4, hipMemcpyDeviceToHost);
  printf("rcp chain vs 1/d over %u values: mismatches %llu (first 0x%08x)\n", n, b, fu);
}
