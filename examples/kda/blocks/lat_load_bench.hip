// Memory latency under load: CTA 0 (1 wave) chases dependent 8 B / lane loads (fresh lines: HBM; or a 2 MB set: L2)
// and times them with s_memtime; CTAs 1..NB stream background loads (4 waves x 8 x 1 KB in flight per CU, bases
// scattered with an odd multiplier to avoid partition camping) for longer than the chase.
#include <hip/hip_runtime.h>
#include <cstdio>
typedef unsigned u2 __attribute__((ext_vector_type(2)));
typedef unsigned u4 __attribute__((ext_vector_type(4)));
__global__ void __launch_bounds__(256) k(const char* g, size_t ws, int nchase, int l2set, int bg_iters, unsigned long long* out, unsigned* sink) {
    const int l = threadIdx.x & 63, w = threadIdx.x >> 6;
    if (blockIdx.x == 0) {
        if (w) return;
        int dep = 0; unsigned x = 0;
        const size_t set = l2set ? ((size_t)2 << 20) : ws;
        unsigned long long t0 = __builtin_amdgcn_s_memtime();
        for (int i = 0; i < nchase; ++i) {
            const size_t off = (((size_t)i * 2654435761u * 4096u) % set) + l * 8 + dep;
            const u2 v = *reinterpret_cast<const u2*>(g + off);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            x ^= v.x; int z = (int)(v.x & 0); asm volatile("" : "+v"(z)); dep = z;
        }
        unsigned long long t1 = __builtin_amdgcn_s_memtime();
        if (l == 0) out[0] = (t1 - t0) / nchase;
        if (x == 0x9e3779b9u) sink[0] = x;
        return;
    }
    const size_t wb = (((size_t)(blockIdx.x * 4 + w) * 2246822519u) % (ws >> 12)) << 12;
    u4 acc = {0, 0, 0, 0};
    for (int it = 0; it < bg_iters; ++it) {
        #pragma unroll
        for (int u = 0; u < 8; ++u) acc ^= *reinterpret_cast<const u4*>(g + ((wb + (size_t)(it * 8 + u) * 1024) % ws) + l * 16);
    }
    if (acc.x == 0x9e3779b9u) sink[blockIdx.x * 256 + threadIdx.x] = acc.y;
}
int main() {
    const size_t ws = (size_t)1 << 30; char* g; unsigned* sink; unsigned long long* o;
    hipMalloc(&g, ws + (1 << 20)); hipMalloc(&sink, 80 * 256 * 4); hipMalloc(&o, 64); hipMemset(g, 0, ws + (1 << 20));
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    for (int l2 = 0; l2 < 2; ++l2) for (int nb : {0, 8, 20, 40, 60, 79}) {
        const int nchase = 2000, bg = nb ? 6000 : 1;
        hipLaunchKernelGGL(k, dim3(1 + nb), dim3(256), 0, 0, g, ws, nchase, l2, bg, o, sink); hipDeviceSynchronize();
        hipEventRecord(a); hipLaunchKernelGGL(k, dim3(1 + nb), dim3(256), 0, 0, g, ws, nchase, l2, bg, o, sink); hipEventRecord(b); hipEventSynchronize(b);
        float ms; hipEventElapsedTime(&ms, a, b); unsigned long long lat; hipMemcpy(&lat, o, 8, hipMemcpyDeviceToHost);
        const double bgbw = nb ? (double)nb * 4 * bg * 8 * 1024 / (ms * 1e-3) / 1e9 : 0;
        printf("LATLOAD target=%s background_cus=%d background_GBs=%.0f latency_cycles=%llu\n", l2 ? "l2" : "hbm", nb, bgbw, lat);
    }
}
