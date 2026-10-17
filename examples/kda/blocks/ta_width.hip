// TA cost of VMEM width at equal bytes: 4 x dword (each coalesced 256 B / wave) vs 1 x dwordx4 (1 KB / wave),
// plus 8 x dword vs 2 x dwordx4, on a buffer larger than L2 (HBM) and one that fits L2.  Reports GB/s.
#include <hip/hip_runtime.h>
#include <cstdio>
__global__ void k_dw(const float* __restrict__ in, float* out, long n4) {   // layout [blk][4][64*4] -> lane reads 4 dwords at stride 256 floats
  long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long w = t >> 6, l = t & 63;
  const float* b = in + w * 1024 + l;
  float s = 0.f;
  #pragma unroll
  for (int r = 0; r < 4; ++r) { const float* bb = b + r * 1024 * gridDim.x * (blockDim.x / 64); s += bb[0] + bb[256] + bb[512] + bb[768]; }
  if (s == 12345.f) out[t] = s;
}
__global__ void k_x4(const float* __restrict__ in, float* out, long n4) {   // layout [blk][64][4] -> lane reads one float4
  long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long w = t >> 6, l = t & 63;
  const float4* b = reinterpret_cast<const float4*>(in + w * 1024) + l;
  float s = 0.f;
  #pragma unroll
  for (int r = 0; r < 4; ++r) { float4 v = *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(b) + r * 1024 * gridDim.x * (blockDim.x / 64)); s += v.x + v.y + v.z + v.w; }
  if (s == 12345.f) out[t] = s;
}
int main() {
  for (long mb : {8L, 1024L}) {
    long nfl = mb * 1024 * 1024 / 4;
    float *in, *out; hipMalloc(&in, nfl * 4); hipMalloc(&out, 64L << 20); hipMemset(in, 0, nfl * 4);
    int threads = 256; long waves = nfl / 1024 / 4; long blocks = waves * 64 / threads;
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    for (int kind = 0; kind < 2; ++kind) {
      float best = 1e30f;
      for (int it = 0; it < 30; ++it) {
        hipEventRecord(a);
        if (kind == 0) hipLaunchKernelGGL(k_dw, dim3(blocks), dim3(threads), 0, 0, in, out, nfl / 4);
        else           hipLaunchKernelGGL(k_x4, dim3(blocks), dim3(threads), 0, 0, in, out, nfl / 4);
        hipEventRecord(b); hipEventSynchronize(b); float ms; hipEventElapsedTime(&ms, a, b); if (it > 3 && ms < best) best = ms;
      }
      printf("%5ld MB  %s : %8.3f us  %7.1f GB/s\n", mb, kind == 0 ? "4 x dword (256B/inst)" : "1 x dwordx4 (1KB/inst)", best * 1e3, nfl * 4 / (best * 1e-3) / 1e9);
    }
    hipFree(in); hipFree(out);
  }
}
