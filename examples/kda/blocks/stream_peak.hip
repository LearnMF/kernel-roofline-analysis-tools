// Saturation points with the best streaming patterns (for the library's sanity check):
//   HBM / L2 read and write: every wave walks its own contiguous region with 16 B / lane (1 KB per instruction),
//   8 independent instructions per iteration, 32 waves / CU; L2: 4 MB working set re-read, HBM: 1 GB.
//   LDS: ds_read_b128 / ds_write_b128 conflict-free, 32 waves / CU.
#include <hip/hip_runtime.h>
#include <cstdio>
typedef unsigned u4 __attribute__((ext_vector_type(4)));
template <int ST>
__global__ void __launch_bounds__(256) k_mem(char* g, size_t ws, int iters, unsigned* out) {
    const size_t nthr = (size_t)gridDim.x * 256, t = (size_t)blockIdx.x * 256 + threadIdx.x;
    u4 acc = {0, 0, 0, 0};
    for (int it = 0; it < iters; ++it) {
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            const size_t off = (((size_t)(it * 8 + u) * nthr + t) * 16) % ws;
            if (ST) *reinterpret_cast<u4*>(g + off) = acc + (unsigned)u; else acc ^= *reinterpret_cast<const u4*>(g + off);
        }
    }
    if (acc.x == 0x9e3779b9u) out[t] = acc.y;
}
__global__ void __launch_bounds__(1024) k_lds(int iters, unsigned* out) {
    __shared__ __align__(16) unsigned char s[32768];
    const unsigned a = (unsigned)(uintptr_t)s + (threadIdx.x & 1023) * 16 % 32768;
    unsigned acc = threadIdx.x;
    for (int it = 0; it < iters; ++it) {
        #pragma unroll
        for (int u = 0; u < 8; ++u) { u4 v; asm volatile("ds_read_b128 %0, %1" : "=v"(v) : "v"(a ^ (u * 4096u))); acc ^= v.x; }
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    }
    if (acc == 0x9e3779b9u) out[threadIdx.x] = acc;
}
template <class F> float best(F f) { hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b); f(); hipDeviceSynchronize(); float m = 1e30f;
    for (int r = 0; r < 7; ++r) { hipEventRecord(a); f(); hipEventRecord(b); hipEventSynchronize(b); float x; hipEventElapsedTime(&x, a, b); m = x < m ? x : m; } return m; }
int main() {
    const size_t big = (size_t)1 << 30; char* g; unsigned* o; hipMalloc(&g, big); hipMalloc(&o, 80 * 8 * 1024 * 4); hipMemset(g, 0, big);
    const int grid = 80 * 8;
    for (size_t ws : {(size_t)4 << 20, big}) for (int st = 0; st < 2; ++st) {
        const int iters = ws == big ? 64 : 256;
        const float ms = best([&] { if (st) hipLaunchKernelGGL(k_mem<1>, dim3(grid), dim3(256), 0, 0, g, ws, iters, o); else hipLaunchKernelGGL(k_mem<0>, dim3(grid), dim3(256), 0, 0, g, ws, iters, o); });
        printf("PEAK %s %s %.3f TB/s\n", ws == big ? "hbm" : "l2", st ? "write" : "read", (double)grid * 256 * iters * 8 * 16 / (ms * 1e-3) / 1e12);
    }
    const int iters = 4000;
    const float ms = best([&] { hipLaunchKernelGGL(k_lds, dim3(160), dim3(1024), 0, 0, iters, o); });
    printf("PEAK lds read_b128 %.3f TB/s (%.1f B/cycle/CU)\n", 160.0 * 1024 * iters * 8 * 16 / (ms * 1e-3) / 1e12, 160.0 * 1024 * iters * 8 * 16 / (ms * 1e-3) / 80 / 1.5e9);
}
