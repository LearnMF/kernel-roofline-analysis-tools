// gfx936: how many waves per SIMD are really co-resident at a given VGPR count?  (The DTK compiler's
// "; Occupancy:" comment assumes 256 VGPRs per SIMD lane -> 1 wave above 128 VGPRs.)  Every wave
// records HW_ID (SE / SH / CU / SIMD / wave slot) and s_memtime at start and end of a fixed busy loop
// that keeps NV floats live; the host counts the peak number of overlapping waves per SIMD.
//   hipcc -O3 --offload-arch=gfx936 occupancy_probe.hip -o occp && ./occp
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <map>

template <int NV>
__global__ void __launch_bounds__(256) k_occ(unsigned long long* rec, int iters) {
    float r[NV];
    #pragma unroll
    for (int i = 0; i < NV; ++i) r[i] = (float)(threadIdx.x + i);
    const unsigned long long t0 = __builtin_amdgcn_s_memtime();
    for (int it = 0; it < iters; ++it) {
        #pragma unroll
        for (int i = 0; i < NV; ++i) r[i] = r[i] * 1.0000001f + 1e-7f;
        asm volatile("" ::: "memory");
    }
    const unsigned long long t1 = __builtin_amdgcn_s_memtime();
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < NV; ++i) s += r[i];
    if ((threadIdx.x & 63) == 0) {
        const unsigned hw = __builtin_amdgcn_s_getreg((31 << 11) | 4);   // HW_REG_HW_ID, 32 bits
        const int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 6;
        rec[w * 3 + 0] = hw; rec[w * 3 + 1] = t0; rec[w * 3 + 2] = t1 + (s == 1234.5f);
    }
}

template <int NV>
static void run(const char* tag, int ctas = 80 * 12) {
    const int waves = ctas * 4;
    unsigned long long* d; hipMalloc(&d, waves * 3 * 8);
    hipLaunchKernelGGL(k_occ<NV>, dim3(ctas), dim3(256), 0, 0, d, 2000);
    hipDeviceSynchronize();
    hipLaunchKernelGGL(k_occ<NV>, dim3(ctas), dim3(256), 0, 0, d, 20000);
    hipDeviceSynchronize();
    std::vector<unsigned long long> h(waves * 3);
    hipMemcpy(h.data(), d, waves * 3 * 8, hipMemcpyDeviceToHost);
    std::map<unsigned, std::vector<std::pair<unsigned long long, int>>> ev;   // per SIMD
    for (int w = 0; w < waves; ++w) {
        const unsigned hw = (unsigned)h[w * 3];
        const unsigned simd = (hw >> 4) & 3, cu = (hw >> 8) & 15, sh = (hw >> 12) & 1, se = (hw >> 13) & 7;
        const unsigned key = (((se * 2 + sh) * 16 + cu) * 4) + simd;
        ev[key].push_back({h[w * 3 + 1], +1});
        ev[key].push_back({h[w * 3 + 2], -1});
    }
    double dur = 0;
    for (int w = 0; w < waves; ++w) dur += (double)(h[w * 3 + 2] - h[w * 3 + 1]);
    printf("%s, %d CTAs: mean wave duration %.0f memtime ticks\n", tag, ctas, dur / waves);
    int peak_all = 0; std::map<int, int> hist;
    for (auto& kv : ev) {
        auto& v = kv.second;
        std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.first < b.first || (a.first == b.first && a.second < b.second); });
        int c = 0, pk = 0;
        for (auto& e : v) { c += e.second; pk = std::max(pk, c); }
        hist[pk]++; peak_all = std::max(peak_all, pk);
    }
    printf("%s: %zu SIMDs seen, peak co-resident waves per SIMD histogram:", tag, ev.size());
    for (auto& kv : hist) printf(" %d->%d", kv.first, kv.second);
    printf("\n");
    hipFree(d);
}

int main() {
    run<230>("NV=230 (256 VGPR), 1 CTA per CU", 80);
    run<230>("NV=230 (256 VGPR), 2 CTA per CU", 160);
    run<24>("NV=24  (~32 VGPR)");
    run<100>("NV=100 (~110 VGPR)");
    run<120>("NV=120 (~128 VGPR)");
    run<160>("NV=160 (~170 VGPR)");
    run<230>("NV=230 (~240 VGPR)");
    return 0;
}
