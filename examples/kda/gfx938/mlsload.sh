#!/bin/bash
source /opt/dtk/env.sh >/dev/null 2>&1; cd /tmp
cat > /tmp/mlsload.hip <<'H'
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
typedef int vec4_int __attribute__((ext_vector_type(4)));
// one wave: matrix_load_32x32_b16 from a row-major global matrix (row stride `stride` passed in rsrc[2]) into LDS, dump LDS
template <int T, int HIBIT>
__global__ void k(const short* g, short* out, int stride_field) {
  __shared__ short s[2048];
  for (int i = threadIdx.x; i < 2048; i += 64) s[i] = -1;
  __syncthreads();
  vec4_int r;
  uint64_t a = (uint64_t)g & 0xffffffffffffull;
  r[0] = (int)(uint32_t)a; r[1] = (int)(uint32_t)(a >> 32); r[2] = stride_field; r[3] = 0;
  r[0] = __builtin_amdgcn_readfirstlane(r[0]); r[1] = __builtin_amdgcn_readfirstlane(r[1]);
  int soff = (int)(size_t)(&s[0]) + (HIBIT ? 0x80000000 : 0);
  __builtin_hcu_matrix_load_32x32_b16(r, (__attribute__((address_space(3))) short*)(soff), 0, T, 0, 0, 0);
  __builtin_amdgcn_s_waitcnt(0);
  __syncthreads();
  for (int i = threadIdx.x; i < 2048; i += 64) out[i] = s[i];
}
int main() {
  const int R = 64, C = 256;   // global matrix 64 x 256 bf16, row-major, element (r, c) = r * 256 + c (fits short for r < 128)
  short h[R * C], o[2048];
  for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c) h[r * C + c] = (short)(r * 256 + c);
  short *dg, *dout; hipMalloc(&dg, sizeof h); hipMalloc(&dout, sizeof o);
  hipMemcpy(dg, h, sizeof h, hipMemcpyHostToDevice);
  auto dump = [&](const char* tag) {
    hipMemcpy(o, dout, sizeof o, hipMemcpyDeviceToHost);
    int filled = 0; for (int i = 0; i < 2048; ++i) filled += o[i] != -1;
    printf("%s: filled %d | LDS[0..7]=", tag, filled);
    for (int i = 0; i < 8; ++i) printf("(%d,%d) ", o[i] / 256, o[i] % 256);
    printf("| LDS[32..35]="); for (int i = 32; i < 36; ++i) printf("(%d,%d) ", o[i] / 256, o[i] % 256);
    printf("| LDS[1023]=(%d,%d)\n", o[1023] / 256, o[1023] % 256);
  };
  for (int sf : {256, 512}) {
    hipLaunchKernelGGL((k<0, 0>), dim3(1), dim3(64), 0, 0, dg, dout, sf); hipDeviceSynchronize(); char t[64]; sprintf(t, "T=0 hi=0 stride=%d", sf); dump(t);
    hipLaunchKernelGGL((k<1, 0>), dim3(1), dim3(64), 0, 0, dg, dout, sf); hipDeviceSynchronize(); sprintf(t, "T=1 hi=0 stride=%d", sf); dump(t);
    hipLaunchKernelGGL((k<0, 1>), dim3(1), dim3(64), 0, 0, dg, dout, sf); hipDeviceSynchronize(); sprintf(t, "T=0 hi=1 stride=%d", sf); dump(t);
  }
  printf("err=%s\n", hipGetErrorString(hipGetLastError()));
}
H
/opt/dtk/bin/hipcc --offload-arch=gfx938 -O3 /tmp/mlsload.hip -o /tmp/mlsload 2>&1 | grep -iE " error" | head -5; HIP_VISIBLE_DEVICES=0 timeout 60 /tmp/mlsload
