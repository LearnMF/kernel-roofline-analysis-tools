// gfx936 probe: does the assembler accept ds_read_m32x16_b16 (and the _alt form), and which LDS
// element lands in which lane / VGPR half?  LDS holds 0..N-1 as b16 (element id = its own index);
// every lane passes address = lane * STRIDE bytes (row-start per lane, like ldmatrix) or a fixed
// base (variant B) and dumps its 4 VGPRs (8 b16).    hipcc -O3 --offload-arch=gfx936 ...
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

template <int MODE>
__global__ void __launch_bounds__(64) probe(uint16_t* out, int stride_bytes) {
    __shared__ __align__(16) uint16_t lds[64 * 64];
    const int l = threadIdx.x;
    for (int i = l; i < 64 * 64; i += 64) lds[i] = (uint16_t)i;
    __syncthreads();
    const uint32_t base = (uint32_t)(uintptr_t)lds;
    const uint32_t addr = base + (MODE & 1 ? 0u : (uint32_t)(l * stride_bytes));
    typedef uint32_t u4 __attribute__((ext_vector_type(4)));
    u4 r;
    if constexpr (MODE < 2) {
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\ts_waitcnt lgkmcnt(0)"
                     : "=v"(r) : "v"(addr) : "memory");
    } else {
        asm volatile("ds_read_m32x16_b16_alt %0, %1 offset:0\n\ts_waitcnt lgkmcnt(0)"
                     : "=v"(r) : "v"(addr) : "memory");
    }
    const uint32_t r0 = r.x, r1 = r.y, r2 = r.z, r3 = r.w;
    out[l * 8 + 0] = r0 & 0xffff; out[l * 8 + 1] = r0 >> 16;
    out[l * 8 + 2] = r1 & 0xffff; out[l * 8 + 3] = r1 >> 16;
    out[l * 8 + 4] = r2 & 0xffff; out[l * 8 + 5] = r2 >> 16;
    out[l * 8 + 6] = r3 & 0xffff; out[l * 8 + 7] = r3 >> 16;
}

int main(int argc, char** argv) {
    const int stride = argc > 1 ? atoi(argv[1]) : 32;
    uint16_t* d; hipMalloc(&d, 64 * 8 * 2);
    uint16_t h[64 * 8];
    const char* names[3] = {"m32x16 per-lane addr", "m32x16 same addr", "m32x16_alt per-lane addr"};
    for (int m = 0; m < 3; ++m) {
        hipMemset(d, 0xff, 64 * 8 * 2);
        if (m == 0) hipLaunchKernelGGL(probe<0>, dim3(1), dim3(64), 0, 0, d, stride);
        if (m == 1) hipLaunchKernelGGL(probe<1>, dim3(1), dim3(64), 0, 0, d, stride);
        if (m == 2) hipLaunchKernelGGL(probe<2>, dim3(1), dim3(64), 0, 0, d, stride);
        const hipError_t e = hipDeviceSynchronize();
        hipMemcpy(h, d, sizeof h, hipMemcpyDeviceToHost);
        printf("== %s (stride %d B): %s\n", names[m], stride, hipGetErrorString(e));
        for (int l = 0; l < 64; ++l) {
            printf("lane %2d:", l);
            for (int k = 0; k < 8; ++k) printf(" %4d", h[l * 8 + k]);
            printf(l % 2 ? "\n" : "   |");
        }
    }
    return 0;
}
