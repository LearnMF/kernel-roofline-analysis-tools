// Does the transcendental unit share the SIMD's VALU issue?  8 waves / SIMD: K waves run v_exp_f32 chains, the
// others run v_fma_f32 chains (8 independent chains each, 16 ops / iteration).  shared pipe -> time ~ sum,
// separate unit -> time ~ max.
#include <hip/hip_runtime.h>
#include <cstdio>
__global__ void __launch_bounds__(256) k(int n, int mode, float* out) {
    const int w = (blockIdx.x * 4 + (threadIdx.x >> 6)) / 80;     // 0..7 for 8 CTAs of 4 waves per CU (grid 640)
    const bool ex = mode == 1 || (mode == 2 && (w & 1));
    const bool none = (mode == 3 && (w & 1)) || (mode == 4 && !(w & 1));
    float x[8];
    for (int c = 0; c < 8; ++c) x[c] = 1.f + threadIdx.x * 1e-4f + c;
    if (!none) for (int it = 0; it < n; ++it) {
        #pragma unroll
        for (int r = 0; r < 2; ++r)
            #pragma unroll
            for (int c = 0; c < 8; ++c) {
                if (ex) asm volatile("v_exp_f32 %0, %0" : "+v"(x[c]));
                else asm volatile("v_fma_f32 %0, %0, %0, 0.5" : "+v"(x[c]));
            }
    }
    float s = 0; for (int c = 0; c < 8; ++c) s += x[c];
    if (s == 1.2345f) out[blockIdx.x] = s;
}
int main() {
    float* o; hipMalloc(&o, 4096); hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    const char* nm[] = {"all fma", "all exp", "half exp + half fma", "half fma only (other half idle)", "half exp only (other half idle)"};
    for (int m = 0; m < 5; ++m) {
        hipLaunchKernelGGL(k, dim3(640), dim3(256), 0, 0, 2000, m, o); hipDeviceSynchronize();
        hipEventRecord(a); hipLaunchKernelGGL(k, dim3(640), dim3(256), 0, 0, 2000, m, o); hipEventRecord(b); hipEventSynchronize(b);
        float ms; hipEventElapsedTime(&ms, a, b); printf("TRANSMIX %-34s %.3f ms\n", nm[m], ms);
    }
}
