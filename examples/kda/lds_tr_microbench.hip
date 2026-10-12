// F2 (layout unification) primitive: throughput of the transposing LDS fragment read ds_read_m32x16_b16
// (lane = row address, gathers a column fragment -- the gfx936 counterpart of ldmatrix.trans, mapping
// probed in round 11) vs the plain fragment read ds_read_b64, at several row pitches (bank conflicts).
// 4 waves per CTA, 2 CTAs per CU, 8 independent reads per iteration, results kept live.
// hipcc -O3 --offload-arch=gfx936 lds_tr_microbench.hip -o ldstr && ./ldstr
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

typedef uint32_t u4 __attribute__((ext_vector_type(4)));
typedef uint32_t u2 __attribute__((ext_vector_type(2)));

template <int MODE>   // 0: ds_read_b64 (8 B/lane), 1: ds_read_b128 (16 B/lane), 2: ds_read_m32x16_b16 (16 B/lane, transposed)
__global__ void __launch_bounds__(256) k_lds(int iters, int pitch_b, uint32_t* out) {
    __shared__ __align__(16) uint16_t lds[16384];          // 32 KB
    const int tid = threadIdx.x, l = tid & 63, w = tid >> 6;
    for (int i = tid; i < 16384; i += 256) lds[i] = (uint16_t)i;
    __syncthreads();
    const uint32_t base = (uint32_t)(uintptr_t)lds + (uint32_t)(w * 4096);
    // per-lane address: row = l & 15 (16 rows per 16-lane group), group g = l >> 4 selects a 16-column block
    const uint32_t addr = base + (uint32_t)((l & 15) * pitch_b + (l >> 4) * 32);
    uint32_t acc = 0;
    for (int it = 0; it < iters; ++it) {
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            const uint32_t a = addr + (uint32_t)(((it + u) & 7) * 16);
            if constexpr (MODE == 0) {
                u2 r; asm volatile("ds_read_b64 %0, %1" : "=v"(r) : "v"(a));
                acc += r.x ^ r.y;
            } else if constexpr (MODE == 1) {
                u4 r; asm volatile("ds_read_b128 %0, %1" : "=v"(r) : "v"(a));
                acc += r.x ^ r.w;
            } else {
                u4 r; asm volatile("ds_read_m32x16_b16 %0, %1" : "=v"(r) : "v"(a));
                acc += r.x ^ r.w;
            }
        }
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    }
    out[blockIdx.x * 256 + tid] = acc;
}

template <int MODE>
static void run(int pitch, uint32_t* out, const char* name) {
    const int grid = 160, iters = 4000;
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    hipLaunchKernelGGL(k_lds<MODE>, dim3(grid), dim3(256), 0, 0, iters, pitch, out);
    hipDeviceSynchronize();
    float best = 1e30f;
    for (int r = 0; r < 5; ++r) {
        hipEventRecord(a);
        hipLaunchKernelGGL(k_lds<MODE>, dim3(grid), dim3(256), 0, 0, iters, pitch, out);
        hipEventRecord(b); hipEventSynchronize(b);
        float ms; hipEventElapsedTime(&ms, a, b); best = ms < best ? ms : best;
    }
    const double insts = (double)grid * 4 * iters * 8;
    const double cyc = best * 1e-3 * 1.5e9 * 80;
    const int bytes = MODE == 0 ? 8 : 16;
    printf("%-22s pitch %4d B: %5.2f CU-cycles per wave instruction, %6.0f GB/s LDS\n", name, pitch, cyc / insts,
           insts * 64 * bytes / (best * 1e-3) / 1e9);
}

int main() {
    uint32_t* out; hipMalloc(&out, 160 * 256 * 4);
    for (int pitch : {256, 272, 288, 512}) {
        run<0>(pitch, out, "ds_read_b64");
        run<1>(pitch, out, "ds_read_b128");
        run<2>(pitch, out, "ds_read_m32x16_b16");
    }
    return 0;
}
