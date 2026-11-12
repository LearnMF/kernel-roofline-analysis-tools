#!/bin/bash
source /opt/dtk/env.sh >/dev/null 2>&1; cd /tmp
cat > /tmp/trmap.hip <<'H'
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
typedef int v4i __attribute__((ext_vector_type(4)));
template <int SWZ, int MODE>
__global__ void k(const short* g, short* out) {
  __shared__ short s[2048];
  v4i r; uint64_t a = (uint64_t)g & 0xffffffffffffull;
  r[0] = __builtin_amdgcn_readfirstlane((int)(uint32_t)a); r[1] = __builtin_amdgcn_readfirstlane((int)(uint32_t)(a >> 32)); r[2] = 64; r[3] = 0;
  const int soff = __builtin_amdgcn_readfirstlane((int)(size_t)&s[0]) + (SWZ ? 0x80000000 : 0);
  __builtin_hcu_matrix_load_32x32_b16(r, (__attribute__((address_space(3))) short*)(soff), 0, 0, 0, 0, 0);
  __builtin_amdgcn_s_waitcnt(0); __syncthreads();
  v4i x; const int m0 = __builtin_amdgcn_readfirstlane((int)(size_t)&s[0]);
  if (MODE == 0) asm volatile("s_mov_b32 m0, %1\n\ts_nop 0\n\tds_read_matrix_trans_format %0, m0 offset:0 element:0x2 row:0x2 col:0x1 alt:0x0\n\ts_waitcnt lgkmcnt(0)" : "=v"(x) : "s"(m0));
  else asm volatile("s_mov_b32 m0, %1\n\ts_nop 0\n\tds_read_matrix_format %0, m0 offset:0 element:0x2 row:0x2 col:0x1 alt:0x0\n\ts_waitcnt lgkmcnt(0)" : "=v"(x) : "s"(m0));
  for (int i = 0; i < 4; ++i) { out[threadIdx.x * 8 + 2 * i] = (short)(x[i] & 0xffff); out[threadIdx.x * 8 + 2 * i + 1] = (short)((uint32_t)x[i] >> 16); }
}
int main() {
  short h[64 * 64], o[512]; for (int r = 0; r < 64; ++r) for (int c = 0; c < 64; ++c) h[r * 64 + c] = (short)(r * 100 + c);   // value = row*100 + col
  short *dg, *dout; hipMalloc(&dg, sizeof h); hipMalloc(&dout, sizeof o); hipMemcpy(dg, h, sizeof h, hipMemcpyHostToDevice);
  auto run = [&](void (*kk)(const short*, short*), const char* nm) {
    hipLaunchKernelGGL(kk, dim3(1), dim3(64), 0, 0, dg, dout); hipDeviceSynchronize(); hipMemcpy(o, dout, sizeof o, hipMemcpyDeviceToHost);
    printf("== %s  (row,col)\n", nm);
    for (int l : {0, 1, 2, 3, 8, 15, 16, 17, 32, 48, 63}) { printf("  L%-2d:", l); for (int i = 0; i < 8; ++i) printf(" (%d,%d)", o[l * 8 + i] / 100, o[l * 8 + i] % 100); printf("\n"); }
  };
  run(k<1, 0>, "MLS swizzled + trans_format r2c1 alt0");
  run(k<0, 1>, "MLS plain + format r2c1 alt0");
}
H
/opt/dtk/bin/hipcc --offload-arch=gfx938 -O3 /tmp/trmap.hip -o /tmp/trmap 2>&1 | grep -E " error" | head; HIP_VISIBLE_DEVICES=0 /tmp/trmap
