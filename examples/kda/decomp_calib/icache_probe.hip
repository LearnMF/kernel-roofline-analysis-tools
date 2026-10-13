// P3 icache probe: 4 wave-specialized code paths (like s4b's A=0..3), each path a
// loop of M independent VALU instructions, total code ~8..128 KB.
// Measures cycles/instruction vs total code bytes; run 1..3 CTAs/CU to test
// whether the L1i is per-CU or shared.
//
//   hipcc -O3 --offload-arch=gfx936 icache_probe.hip -o icache_probe
//   ./icache_probe <M> <grid_ctas>     # one config per run
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#ifndef M_UNROLL
#define M_UNROLL 64
#endif
#ifndef N_ITER
#define N_ITER 400
#endif

__global__ void __launch_bounds__(256) icache_probe(float* out, int sel) {
    int w = threadIdx.x / 64;
    float a = threadIdx.x + 1.5f, b = a + 1.f, c = a + 2.f, d = a + 3.f;
    // 4 disjoint paths, each with M_UNROLL independent VALU ops inside a loop.
    switch ((w + sel) & 3) {
        case 0:
            for (int it = 0; it < N_ITER; ++it) {
#pragma unroll
                for (int u = 0; u < M_UNROLL; ++u) a = fmaf(a, 1.0000001f, b);
            }
            break;
        case 1:
            for (int it = 0; it < N_ITER; ++it) {
#pragma unroll
                for (int u = 0; u < M_UNROLL; ++u) b = fmaf(b, 1.0000002f, c);
            }
            break;
        case 2:
            for (int it = 0; it < N_ITER; ++it) {
#pragma unroll
                for (int u = 0; u < M_UNROLL; ++u) c = fmaf(c, 1.0000003f, d);
            }
            break;
        default:
            for (int it = 0; it < N_ITER; ++it) {
#pragma unroll
                for (int u = 0; u < M_UNROLL; ++u) d = fmaf(d, 1.0000004f, a);
            }
            break;
    }
    if (a + b + c + d == 12345.f) out[threadIdx.x] = a;
}

int main(int argc, char** argv) {
    int m = argc > 1 ? atoi(argv[1]) : M_UNROLL;
    int g = argc > 2 ? atoi(argv[2]) : 80;
    (void)m;
    float* o; hipMalloc(&o, 1024);
    for (int r = 0; r < 3; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        hipLaunchKernelGGL(icache_probe, dim3(g), dim3(256), 0, 0, o, r);
        hipDeviceSynchronize();
        double us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - t0).count();
        printf("r%d %.1f us\n", r, us);
    }
    return 0;
}
