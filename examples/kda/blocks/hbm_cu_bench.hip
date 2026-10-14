// HBM bandwidth per CU vs the number of active CUs (1 CTA per CU, grid = k <= 80): each wave streams fresh 1 KB
// rows (contiguous, 16 B / lane) or 16-row x 64 B fragments, 8 independent loads per iteration.
#include <hip/hip_runtime.h>
#include <cstdio>
typedef unsigned u4 __attribute__((ext_vector_type(4)));
template <int FRAG>
__global__ void k(const char* g, size_t ws, int iters, unsigned* out) {
    const int l = threadIdx.x & 63, w = threadIdx.x >> 6, nw = blockDim.x >> 6;
    const size_t lo = FRAG ? (size_t)(l & 15) * 24576 + (l >> 4) * 16 : (size_t)l * 16;
    const size_t span = FRAG ? 64 : 1024, wb = ((size_t)(blockIdx.x * nw + w) * 1048576 * 7) % ws;
    u4 acc = {0, 0, 0, 0};
    for (int it = 0; it < iters; ++it) {
        #pragma unroll
        for (int u = 0; u < 8; ++u) acc ^= *reinterpret_cast<const u4*>(g + ((wb + (size_t)(it * 8 + u) * (FRAG ? 393216 : span)) % ws) + lo);
    }
    if (acc.x == 0x9e3779b9u) out[blockIdx.x * 1024 + threadIdx.x] = acc.y;
}
int main() {
    const size_t ws = (size_t)1 << 30; char* g; unsigned* o; hipMalloc(&g, ws + (1 << 24)); hipMalloc(&o, 80 * 1024 * 4); hipMemset(g, 0, ws + (1 << 24));
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    for (int frag = 0; frag < 2; ++frag) for (int thr : {256, 1024}) for (int kc : {1, 2, 4, 8, 16, 32, 48, 64, 80}) {
        const int iters = 256;
        auto L = [&] { if (frag) hipLaunchKernelGGL(k<1>, dim3(kc), dim3(thr), 0, 0, g, ws, iters, o); else hipLaunchKernelGGL(k<0>, dim3(kc), dim3(thr), 0, 0, g, ws, iters, o); };
        L(); hipDeviceSynchronize(); float best = 1e30f;
        for (int r = 0; r < 3; ++r) { hipEventRecord(a); L(); hipEventRecord(b); hipEventSynchronize(b); float ms; hipEventElapsedTime(&ms, a, b); best = ms < best ? ms : best; }
        const double bytes = (double)kc * (thr / 64) * iters * 8 * 1024;   // requested bytes (fragments fetch 2x lines)
        printf("HBMCU pattern=%s waves_per_cu=%d active_cus=%d GBs_total=%.0f GBs_per_cu=%.1f\n", frag ? "16x64B" : "contig1KB", thr / 64, kc, bytes / (best * 1e-3) / 1e9, bytes / (best * 1e-3) / 1e9 / kc);
    }
}
