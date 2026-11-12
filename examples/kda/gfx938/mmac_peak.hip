// bf16 MMAC peak on gfx938: 8 independent accumulators per wave, v_mmac_f32_16x16x16_bf16 (16*16*16*2 flop each)
#include <hip/hip_runtime.h>
#include <cstdio>
typedef short v4s __attribute__((ext_vector_type(4)));
typedef float f4 __attribute__((ext_vector_type(4)));
__global__ void k(float* o, int it) {
  v4s a = {(short)threadIdx.x, 1, 2, 3}, b = {3, 2, 1, (short)threadIdx.x};
  f4 c[8]; for (int i = 0; i < 8; ++i) c[i] = f4{0, 0, 0, (float)i};
  for (int r = 0; r < it; ++r) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) c[i] = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(a, b, c[i], 0, 0);
  }
  float s = 0; for (int i = 0; i < 8; ++i) s += c[i][0] + c[i][3];
  o[blockIdx.x * 256 + threadIdx.x] = s;
}
int main() {
  float* o; hipMalloc(&o, 1 << 24); hipDeviceProp_t pr; hipGetDeviceProperties(&pr, 0);
  hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1); float ms; const int it = 20000;
  for (int cta_per_cu : {1, 2, 4}) {
    const int g = pr.multiProcessorCount * cta_per_cu;
    k<<<g, 256>>>(o, it); hipEventRecord(e0); k<<<g, 256>>>(o, it); hipEventRecord(e1); hipEventSynchronize(e1);
    hipEventElapsedTime(&ms, e0, e1);
    const double fl = (double)g * 4 * it * 8 * 16 * 16 * 16 * 2;
    printf("%s  %d CTA/CU  bf16 MMAC %.1f TFLOP/s\n", pr.name, cta_per_cu, fl / (ms * 1e9));
  }
}
