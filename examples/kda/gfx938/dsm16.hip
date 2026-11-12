// ds_read_m32x16_b16 semantics probe: LDS = index pattern, per-lane random 16 B-aligned addresses, dump 8 shorts / lane
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
typedef short v8s __attribute__((ext_vector_type(8)));
__global__ void k(const int* addr, short* out) {
  __shared__ __align__(16) short s[8192];
  for (int i = threadIdx.x; i < 8192; i += 64) s[i] = (short)(i * 7 + 3);
  __syncthreads();
  const short* p = s + addr[blockIdx.x * 64 + threadIdx.x] * 8;   // 16 B granules
  v8s r = __builtin_hcu_ds_read_m32x16_bf16((short*)p);
  for (int i = 0; i < 8; ++i) out[(blockIdx.x * 64 + threadIdx.x) * 8 + i] = r[i];
}
int main() {
  const int NB = 64; int h_addr[NB * 64]; uint32_t x = 12345;
  for (int b = 0; b < NB; ++b) for (int l = 0; l < 64; ++l) { x = x * 1664525u + 1013904223u; h_addr[b * 64 + l] = b == 0 ? l : (int)((x >> 8) % 1024); }
  int* da; short* dout; hipMalloc(&da, sizeof(h_addr)); hipMalloc(&dout, NB * 64 * 8 * 2);
  hipMemcpy(da, h_addr, sizeof(h_addr), hipMemcpyHostToDevice);
  hipLaunchKernelGGL(k, dim3(NB), dim3(64), 0, 0, da, dout); printf("err %s\n", hipGetErrorString(hipDeviceSynchronize()));
  static short h[NB * 64 * 8]; hipMemcpy(h, dout, sizeof(h), hipMemcpyDeviceToHost);
  FILE* f = fopen("/tmp/dsm16.bin", "wb"); fwrite(h, 1, sizeof(h), f); fclose(f);
  printf("block0 (lane l addresses granule l):\n");
  for (int l : {0, 1, 7, 8, 9, 15, 16, 17, 63}) { printf(" L%-2d:", l); for (int i = 0; i < 8; ++i) printf(" %6d", h[l * 8 + i]); printf("\n"); }
}
