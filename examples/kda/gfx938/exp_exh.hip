// Exhaustive: expf without the overflow / underflow selects (the ocml sequence of the prep_a ISA) vs expf, for every
// fp32 with |x| <= 87 and every NaN; rcp chain vs 1/d for every NaN d.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
__device__ __forceinline__ float expf_core(float x) {
  #pragma clang fp contract(off)
  const float ph = x * __uint_as_float(0x3fb8aa3bu);
  float pl = __builtin_fmaf(x, __uint_as_float(0x3fb8aa3bu), -ph);
  const float e = __builtin_rintf(ph);
  pl = __builtin_fmaf(x, __uint_as_float(0x32a5705fu), pl);
  const float t = (ph - e) + pl;
  return __builtin_amdgcn_ldexpf(__builtin_amdgcn_exp2f(t), (int)e);
}
__device__ __forceinline__ float rcp1(float d) {
  #pragma clang fp contract(off)
  const float r0 = __builtin_amdgcn_rcpf(d);
  const float r1 = __builtin_fmaf(__builtin_fmaf(-d, r0, 1.f), r0, r0);
  const float q1 = __builtin_fmaf(__builtin_fmaf(-d, r1, 1.f), r1, r1);
  return __builtin_fmaf(__builtin_fmaf(-d, q1, 1.f), r1, q1);
}
__global__ void k(uint32_t base, uint32_t n, unsigned long long* bad, uint32_t* first, int mode) {
  const uint32_t i = blockIdx.x * 256u + threadIdx.x; if (i >= n) return;
  const uint32_t u = base + i; const float x = __uint_as_float(u);
  volatile float one = 1.f;
  bool ok;
  if (mode == 0) ok = __float_as_uint(expf(x)) == __float_as_uint(expf_core(x));
  else ok = __float_as_uint(one / x) == __float_as_uint(rcp1(x));
  if (u == 0x3f000000u && mode == 0) ok = false;   // canary
  if (!ok) { atomicAdd(bad, 1ull); atomicMin(first, u); }
}
static void run(const char* nm, uint32_t lo, uint32_t hi, int mode) {   // [lo, hi] inclusive
  unsigned long long* bad; uint32_t* first; hipMalloc(&bad, 8); hipMalloc(&first, 4);
  hipMemset(bad, 0, 8); uint32_t f0 = 0xffffffffu; hipMemcpy(first, &f0, 4, hipMemcpyHostToDevice);
  const unsigned long long n = (unsigned long long)hi - lo + 1;
  for (unsigned long long off = 0; off < n; off += 256ull * 65536ull) {
    const uint32_t m = (uint32_t)((n - off) < 256ull * 65536ull ? (n - off) : 256ull * 65536ull);
    hipLaunchKernelGGL(k, dim3((m + 255) / 256), dim3(256), 0, 0, (uint32_t)(lo + off), m, bad, first, mode);
  }
  hipError_t e = hipDeviceSynchronize();
  unsigned long long b; uint32_t fu; hipMemcpy(&b, bad, 8, hipMemcpyDeviceToHost); hipMemcpy(&fu, first, 4, hipMemcpyDeviceToHost);
  printf("%-34s %s  values %llu  mismatches %llu (first 0x%08x)\n", nm, hipGetErrorString(e), n, b, fu);
}
int main() {
  const uint32_t b87 = 0x42ae0000u;                     // 87.0f
  run("expf_core, x in [+0, 87]", 0x00000000u, b87, 0);  // includes the canary 0.5
  run("expf_core, x in [-0, -87]", 0x80000000u, 0x80000000u | b87, 0);
  run("expf_core, +NaN", 0x7f800001u, 0x7fffffffu, 0);
  run("expf_core, -NaN", 0xff800001u, 0xffffffffu, 0);
  run("rcp chain, +NaN", 0x7f800001u, 0x7fffffffu, 1);
  run("rcp chain, -NaN", 0xff800001u, 0xffffffffu, 1);
}
