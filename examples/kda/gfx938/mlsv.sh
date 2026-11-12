#!/bin/bash
source /opt/dtk/env.sh >/dev/null 2>&1; cd /tmp
echo "family:"; grep -E "ds_(s_)?read_m|ds_(s_)?write_m" /tmp/hcu_builtins.txt | tr '\n' ' '; echo
cat > /tmp/mlsv.hip <<'H'
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
typedef short short8_t __attribute__((ext_vector_type(8)));
__global__ void k(short* out, const short* lds_init, const int* addr, int variant) {
  __shared__ short s[4096];
  for (int i = threadIdx.x; i < 4096; i += 64) s[i] = lds_init[i];
  __syncthreads();
  short* p = &s[addr[threadIdx.x]];
  short8_t r;
  if (variant == 0) r = __builtin_hcu_ds_read_m32x16_bf16(p);
  else r = __builtin_hcu_ds_read_m32x16_bf16_alt(p);
  for (int i = 0; i < 8; ++i) out[threadIdx.x * 8 + i] = r[i];
}
int main() {
  short h_lds[4096], h_out[512]; int h_addr[64];
  for (int i = 0; i < 4096; ++i) h_lds[i] = (short)(i * 7 + 3);
  srand(1);
  short *d_lds, *d_out; int* d_addr;
  hipMalloc(&d_lds, sizeof h_lds); hipMalloc(&d_out, sizeof h_out); hipMalloc(&d_addr, sizeof h_addr);
  hipMemcpy(d_lds, h_lds, sizeof h_lds, hipMemcpyHostToDevice);
  const int S[8] = {0, 4, 8, 12, 2, 6, 10, 14};
  for (int variant = 0; variant < 2; ++variant) {
    int bad = 0, trials = 50;
    for (int t = 0; t < trials; ++t) {
      for (int l = 0; l < 64; ++l) h_addr[l] = (rand() % 504) * 8;   // 16B-aligned rows
      hipMemcpy(d_addr, h_addr, sizeof h_addr, hipMemcpyHostToDevice);
      hipLaunchKernelGGL(k, dim3(1), dim3(64), 0, 0, d_out, d_lds, d_addr, variant);
      hipMemcpy(h_out, d_out, sizeof h_out, hipMemcpyDeviceToHost);
      for (int l = 0; l < 64; ++l) {
        const int g = l / 16, j = l % 16;
        for (int i = 0; i < 8; ++i) {
          const int src = g * 16 + (j / 8) + S[i];
          if (h_out[l * 8 + i] != h_lds[h_addr[src] + (j % 8)]) ++bad;
        }
      }
      if (t == 0 && variant == 1) {   // print a sample to derive the alt mapping
        printf("alt sample (lane: values as (srcLane,pos)):\n");
        for (int l : {0, 1, 7, 8, 9, 15}) { printf(" L%d:", l);
          for (int i = 0; i < 8; ++i) { int v = h_out[l * 8 + i]; int found = -1, pos = -1;
            for (int s2 = 0; s2 < 16 && found < 0; ++s2) for (int q = 0; q < 8; ++q) if (h_lds[h_addr[s2] + q] == v) { found = s2; pos = q; break; }
            printf(" (%d,%d)", found, pos); }
          printf("\n"); }
      }
    }
    printf("variant %s: model mismatches %d of %d\n", variant ? "m32x16_bf16_alt" : "m32x16_bf16", bad, trials * 512);
  }
}
H
/opt/dtk/bin/hipcc --offload-arch=gfx938 -O3 /tmp/mlsv.hip -o /tmp/mlsv 2>&1 | grep -i " error" | head -3; HIP_VISIBLE_DEVICES=0 /tmp/mlsv
