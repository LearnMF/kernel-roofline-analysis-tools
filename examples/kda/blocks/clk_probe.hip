// shader clock under load: s_memtime ticks vs hipEvent wall time (and the s_memrealtime rate) on a busy GPU
#include <hip/hip_runtime.h>
#include <cstdio>
__global__ void k(unsigned long long* o, int n) {
    float a = threadIdx.x;
    unsigned long long t0 = __builtin_amdgcn_s_memtime(), r0 = __builtin_amdgcn_s_memrealtime();
    for (int i = 0; i < n; ++i) a = a * 1.0000001f + 0.5f;
    unsigned long long t1 = __builtin_amdgcn_s_memtime(), r1 = __builtin_amdgcn_s_memrealtime();
    if (threadIdx.x == 0) { o[blockIdx.x * 2] = t1 - t0; o[blockIdx.x * 2 + 1] = r1 - r0; }
    if (a == 1.2345f) o[0] = 0;
}
int main() {
    unsigned long long* d; hipMalloc(&d, 80 * 16);
    unsigned long long h[160];
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    for (int r = 0; r < 3; ++r) {
        hipEventRecord(e0);
        hipLaunchKernelGGL(k, dim3(80), dim3(256), 0, 0, d, 20000000);   // one CTA per CU, all start together
        hipEventRecord(e1); hipEventSynchronize(e1);
        float ms; hipEventElapsedTime(&ms, e0, e1);
        hipMemcpy(h, d, 80 * 16, hipMemcpyDeviceToHost);
        double mt = 0, rt = 0; for (int b = 0; b < 80; ++b) { mt += h[2 * b]; rt += h[2 * b + 1]; }
        mt /= 80; rt /= 80;
        printf("wall %.3f ms: s_memtime %.4g ticks -> %.1f MHz ; s_memrealtime %.4g ticks -> %.2f MHz\n",
               ms, mt, mt / (ms * 1e3), rt, rt / (ms * 1e3));
    }
}
