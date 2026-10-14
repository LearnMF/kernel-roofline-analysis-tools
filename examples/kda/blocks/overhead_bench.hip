// Building-block library P2: kernel-level overheads on gfx936.
//   LAUNCH : N back-to-back empty kernels on one stream (dependent launches) -> time per launch
//   MIN    : one empty kernel (hipEvent around it) and one kernel doing a single 1 KB load + store per CTA
//   TAIL   : fixed per-CTA work (~20 us of dependent FMAs, 256 threads, 1 CTA / CU by LDS), grid = 80 .. 401
//            -> time steps by rounds = ceil(ctas / 80)
//   DISPATCH: very short CTAs (64 threads, 32 FMAs), grid 80 x k -> CTA launch rate per CU
#include <hip/hip_runtime.h>
#include <cstdio>
__global__ void k_empty() {}
__global__ void k_min(const float* a, float* b) { b[blockIdx.x * 256 + threadIdx.x] = a[blockIdx.x * 256 + threadIdx.x] + 1.f; }
__global__ void __launch_bounds__(256) k_work(int n, float* out) {
    extern __shared__ float s[];
    float x = threadIdx.x;
    for (int i = 0; i < n; ++i) x = x * 1.0000001f + 0.5f;
    if (x == 1.2345f) { s[0] = x; out[blockIdx.x] = s[threadIdx.x]; }
}
__global__ void __launch_bounds__(64) k_tiny(float* out) {
    float x = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < 32; ++i) x = x * 1.0000001f + 0.5f;
    if (x == 1.2345f) out[blockIdx.x] = x;
}
template <class F> float ms(F f, int reps = 5) {
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    f(); hipDeviceSynchronize();
    float best = 1e30f;
    for (int r = 0; r < reps; ++r) { hipEventRecord(a); f(); hipEventRecord(b); hipEventSynchronize(b); float m; hipEventElapsedTime(&m, a, b); best = m < best ? m : best; }
    return best;
}
int main() {
    float *a, *b; hipMalloc(&a, 401 * 256 * 4); hipMalloc(&b, 401 * 256 * 4);
    for (int n : {1, 2, 8, 32, 128}) {
        const float t = ms([&] { for (int i = 0; i < n; ++i) hipLaunchKernelGGL(k_empty, dim3(1), dim3(64), 0, 0); });
        printf("LAUNCH n=%d us_per_launch=%.2f\n", n, t * 1e3 / n);
    }
    printf("MIN empty_us=%.2f one_load_store_80ctas_us=%.2f\n", ms([&] { hipLaunchKernelGGL(k_empty, dim3(1), dim3(64), 0, 0); }) * 1e3,
           ms([&] { hipLaunchKernelGGL(k_min, dim3(80), dim3(256), 0, 0, a, b); }) * 1e3);
    const int n = 20000;
    for (int g : {80, 81, 120, 160, 161, 240, 241, 320, 400, 401}) {
        const float t = ms([&] { hipLaunchKernelGGL(k_work, dim3(g), dim3(256), 40000, 0, n, b); });   // 40 KB LDS -> 1 CTA / CU
        printf("TAIL ctas=%d us=%.2f rounds=%d\n", g, t * 1e3, (g + 79) / 80);
    }
    for (int k : {1, 8, 64, 512}) {
        const float t = ms([&] { hipLaunchKernelGGL(k_tiny, dim3(80 * k), dim3(64), 0, 0, b); });
        printf("DISPATCH ctas=%d us=%.2f ns_per_cta_per_cu=%.1f\n", 80 * k, t * 1e3, t * 1e6 / k);
    }
}
