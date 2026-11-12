#!/bin/bash
source /opt/dtk/env.sh >/dev/null 2>&1; cd /tmp
cat > /tmp/dsm.hip <<'H'
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
typedef int v4i __attribute__((ext_vector_type(4)));
// LDS holds 2048 shorts with value = index; read one ds_read_matrix variant at m0 = base, dump the 4 dwords (8 shorts) per lane
#define RD(NAME, MODS) \
__global__ void NAME(short* out) { \
  __shared__ short s[4096]; \
  for (int i = threadIdx.x; i < 4096; i += 64) s[i] = (short)i; \
  __syncthreads(); \
  v4i r; int base = (int)(size_t)&s[0]; \
  asm volatile("s_mov_b32 m0, %1\n\ts_nop 0\n\t" MODS "\n\ts_waitcnt lgkmcnt(0)" : "=v"(r) : "s"(base)); \
  for (int i = 0; i < 4; ++i) { out[threadIdx.x * 8 + 2 * i] = (short)(r[i] & 0xffff); out[threadIdx.x * 8 + 2 * i + 1] = (short)((uint32_t)r[i] >> 16); } }
RD(k_fmt_a0, "ds_read_matrix_format %0, m0 offset:0 element:0x2 row:0x2 col:0x1 alt:0x0")
RD(k_fmt_a1, "ds_read_matrix_format %0, m0 offset:0 element:0x2 row:0x2 col:0x1 alt:0x1")
RD(k_tr_a0,  "ds_read_matrix_trans_format %0, m0 offset:0 element:0x2 row:0x2 col:0x1 alt:0x0")
RD(k_tr_a1,  "ds_read_matrix_trans_format %0, m0 offset:0 element:0x2 row:0x1 col:0x2 alt:0x1")
int main() {
  short* d; hipMalloc(&d, 1024); short h[512];
  void (*ks[4])(short*) = {k_fmt_a0, k_fmt_a1, k_tr_a0, k_tr_a1};
  const char* nm[4] = {"format r2c1 alt0", "format r2c1 alt1", "trans r2c1 alt0", "trans r1c2 alt1"};
  for (int v = 0; v < 4; ++v) {
    hipLaunchKernelGGL(ks[v], dim3(1), dim3(64), 0, 0, d); hipError_t e = hipDeviceSynchronize();
    hipMemcpy(h, d, 1024, hipMemcpyDeviceToHost);
    int mx = 0; for (int i = 0; i < 512; ++i) mx = h[i] > mx ? h[i] : mx;
    printf("== %s (err %s, max idx %d): idx as (row,col) of a 32-wide row-major tile\n", nm[v], hipGetErrorString(e), mx);
    for (int l : {0, 1, 2, 15, 16, 17, 31, 32, 48, 63}) { printf("  L%-2d:", l); for (int i = 0; i < 8; ++i) printf(" %4d", h[l * 8 + i]); printf("\n"); }
  }
}
H
/opt/dtk/bin/hipcc --offload-arch=gfx938 -O3 /tmp/dsm.hip -o /tmp/dsm 2>&1 | grep -iE " error|error:" | head -5; HIP_VISIBLE_DEVICES=0 timeout 60 /tmp/dsm
