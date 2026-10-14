// Issue-side buffering of VMEM / LDS on gfx936, read from SQTT (hipprof --sqtt --sqtt-type 1): each wave issues
// bursts of N back-to-back instructions (production forms), waits for all of them, repeats R times.  Data is warmed
// into L2 first (4 MB region, every burst touches new lines so the per-CU L1 does not hit).  The SQTT issue time of
// every instruction in a burst shows where issue starts to block: in-flight limit per wave (1 wave / CU) and per CU
// (2 / 4 waves on different SIMDs bursting together).
//   KIND 0 global_load_dwordx2 (512 B rows)   1 global_load_dwordx4 (1 KB)   2 ds_read2_b64   3 ds_write_b16_d16_hi
//   hipcc -O3 --offload-arch=gfx936 issue_burst_bench.hip -o ibb
//   hipprof --sqtt --sqtt-type 1 --kernel-name k_burst --output-type 0 -o <dir>/x ./ibb <kind> <N> <waves>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
typedef unsigned u2 __attribute__((ext_vector_type(2)));
typedef unsigned u4 __attribute__((ext_vector_type(4)));

__global__ void k_warm(const char* g, size_t n, unsigned* out) {
    unsigned x = 0;
    for (size_t i = (blockIdx.x * blockDim.x + threadIdx.x) * 16; i < n; i += (size_t)gridDim.x * blockDim.x * 16)
        x ^= *reinterpret_cast<const unsigned*>(g + i);
    if (x == 0x9e3779b9u) out[0] = x;
}

template <int KIND, int N>
__global__ void k_burst(const char* __restrict__ g, int R, unsigned* out) {
    __shared__ __align__(16) unsigned s[16384];
    const int l = threadIdx.x & 63, w = threadIdx.x >> 6;
    const unsigned a = (unsigned)(uintptr_t)(s + (w & 3) * 4096 + l * 2);
    unsigned x = 0;
    for (int r = 0; r < R; ++r) {
        const char* p = g + (((size_t)blockIdx.x * 8 + w) * 16 + (size_t)r) * 32768 % ((size_t)4 << 20) + l * (KIND == 1 ? 16 : 8);
        if (KIND == 0) {
            u2 v[N];
            #pragma unroll
            for (int i = 0; i < N; ++i) v[i] = *reinterpret_cast<const u2*>(p + i * 1024);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            #pragma unroll
            for (int i = 0; i < N; ++i) x ^= v[i].x;
        } else if (KIND == 1) {
            u4 v[N];
            #pragma unroll
            for (int i = 0; i < N; ++i) v[i] = *reinterpret_cast<const u4*>(p + i * 1024);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            #pragma unroll
            for (int i = 0; i < N; ++i) x ^= v[i].x;
        } else if (KIND == 2) {
            u4 q[N];
            #pragma unroll
            for (int i = 0; i < N; ++i) asm volatile("ds_read2_b64 %0, %1 offset1:4" : "=v"(q[i]) : "v"(a + (unsigned)(i & 7) * 1024u));
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
            #pragma unroll
            for (int i = 0; i < N; ++i) x ^= q[i].x;
        } else {
            #pragma unroll
            for (int i = 0; i < N; ++i) asm volatile("ds_write_b16_d16_hi %0, %1" :: "v"(a + (unsigned)(i & 7) * 1024u), "v"(x + i));
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        }
    }
    if (x == 0x9e3779b9u) out[blockIdx.x * 256 + threadIdx.x] = x;
}

template <int KIND>
static void launch(int N, int waves, const char* g, unsigned* out) {
    const int R = 8;
#define C(n) case n: hipLaunchKernelGGL((k_burst<KIND, n>), dim3(80), dim3(64 * waves), 0, 0, g, R, out); break;
    switch (N) { C(4) C(8) C(12) C(16) C(24) C(32) default: printf("bad N\n"); }
#undef C
}

int main(int argc, char** argv) {
    const int kind = atoi(argv[1]), N = atoi(argv[2]), waves = atoi(argv[3]);
    const size_t ws = (size_t)4 << 20;
    char* g; unsigned* out;
    hipMalloc(&g, ws + (1 << 20)); hipMalloc(&out, 80 * 256 * 4); hipMemset(g, 0, ws + (1 << 20));
    hipLaunchKernelGGL(k_warm, dim3(320), dim3(256), 0, 0, g, ws, out);   // L2-resident afterwards
    hipLaunchKernelGGL(k_warm, dim3(320), dim3(256), 0, 0, g, ws, out);
    if (kind == 0) launch<0>(N, waves, g, out);
    if (kind == 1) launch<1>(N, waves, g, out);
    if (kind == 2) launch<2>(N, waves, g, out);
    if (kind == 3) launch<3>(N, waves, g, out);
    hipDeviceSynchronize();
    printf("done kind=%d N=%d waves=%d\n", kind, N, waves);
    return 0;
}
