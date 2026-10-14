// Issue-side buffering: s_memtime around a burst of N back-to-back VMEM loads (8 B/lane, 512 B rows) or LDS
// ds_read2_b64, without waiting for completion; 1 wave / CU and 4 waves / CU (one per SIMD, all bursting together).
#include <hip/hip_runtime.h>
#include <cstdio>
typedef unsigned u2 __attribute__((ext_vector_type(2)));
typedef unsigned u4 __attribute__((ext_vector_type(4)));
template <int N, int LDS>
__global__ void k(const char* g, unsigned long long* out) {
    __shared__ __align__(16) unsigned s[8192];
    const int l = threadIdx.x & 63, w = threadIdx.x >> 6;
    const char* p = g + (size_t)blockIdx.x * 65536 + w * 16384 + l * 8;
    const unsigned a = (unsigned)(uintptr_t)(s + w * 2048 + l * 2);
    u2 v[N]; u4 q[N];
    __syncthreads();
    unsigned long long t0 = __builtin_amdgcn_s_memtime();
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        if (LDS) asm volatile("ds_read2_b64 %0, %1 offset1:4" : "=v"(q[i]) : "v"(a + i * 512u));
        else v[i] = *reinterpret_cast<const u2*>(p + i * 512);
    }
    unsigned long long t1 = __builtin_amdgcn_s_memtime();
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
    unsigned x = 0;
    #pragma unroll
    for (int i = 0; i < N; ++i) x ^= LDS ? q[i].x : v[i].x;
    if (l == 0) out[blockIdx.x * 4 + w] = t1 - t0;
    if (x == 0x9e3779b9u) out[0] = x;
}
template <int N, int LDS>
void run(const char* g, unsigned long long* d, unsigned long long* h) {
    for (int wv : {1, 4}) {
        double s = 0; int c = 0;
        for (int r = 0; r < 5; ++r) {
            hipLaunchKernelGGL((k<N, LDS>), dim3(80), dim3(64 * wv), 0, 0, g, d);
            hipMemcpy(h, d, 80 * 4 * 8, hipMemcpyDeviceToHost);
            for (int b = 0; b < 80; ++b) for (int w = 0; w < wv; ++w) { s += h[b * 4 + w]; ++c; }
        }
        printf("BURST %s N=%2d waves_per_cu=%d issue_cycles=%.0f per_inst=%.1f\n", LDS ? "ds_read2_b64" : "global_load_dwordx2", N, wv, s / c, s / c / N);
    }
}
int main() {
    char* g; unsigned long long *d, h[320];
    hipMalloc(&g, 80 << 16); hipMalloc(&d, 320 * 8); hipMemset(g, 0, 80 << 16);
    run<1, 0>(g, d, h); run<4, 0>(g, d, h); run<8, 0>(g, d, h); run<16, 0>(g, d, h); run<24, 0>(g, d, h);
    run<1, 1>(g, d, h); run<4, 1>(g, d, h); run<8, 1>(g, d, h); run<16, 1>(g, d, h);
}
