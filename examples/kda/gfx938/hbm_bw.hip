// Achievable HBM bandwidth on this card: streaming read (sum), streaming copy, 16 B per lane, 1 GB working set.
#include <hip/hip_runtime.h>
#include <cstdio>
typedef float f4 __attribute__((ext_vector_type(4)));
__global__ void rd(const f4* __restrict__ a, size_t n, float* o) {
  f4 s = {0, 0, 0, 0};
  for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += (size_t)gridDim.x * 256) s += a[i];
  if (s[0] + s[1] + s[2] + s[3] == 1234.5f) o[0] = 1;
}
__global__ void cp(const f4* __restrict__ a, f4* __restrict__ b, size_t n) {
  for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += (size_t)gridDim.x * 256) b[i] = a[i];
}
int main() {
  const size_t bytes = 1ull << 30, n = bytes / 16; f4 *a, *b; float* o;
  hipMalloc(&a, bytes); hipMalloc(&b, bytes); hipMalloc(&o, 4); hipMemset(a, 0, bytes);
  hipDeviceProp_t pr; hipGetDeviceProperties(&pr, 0);
  hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1); float ms;
  for (int g : {pr.multiProcessorCount * 4, pr.multiProcessorCount * 8, pr.multiProcessorCount * 16}) {
    rd<<<g, 256>>>(a, n, o); hipEventRecord(e0); for (int i = 0; i < 5; ++i) rd<<<g, 256>>>(a, n, o); hipEventRecord(e1); hipEventSynchronize(e1);
    hipEventElapsedTime(&ms, e0, e1); printf("%s grid %5d  read  %.0f GB/s\n", pr.name, g, 5 * bytes / (ms * 1e6));
    cp<<<g, 256>>>(a, b, n); hipEventRecord(e0); for (int i = 0; i < 5; ++i) cp<<<g, 256>>>(a, b, n); hipEventRecord(e1); hipEventSynchronize(e1);
    hipEventElapsedTime(&ms, e0, e1); printf("%s grid %5d  copy  %.0f GB/s (read + write)\n", pr.name, g, 5 * 2.0 * bytes / (ms * 1e6));
  }
}
