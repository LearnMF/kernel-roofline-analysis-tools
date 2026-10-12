// VMEM-issue (TA) cost vs access pattern on gfx936 -- the model term kda_bound / design_eval lacked
// (P-45, R36).  Each wave issues 8 independent VMEM instructions per iteration with one of the access
// patterns the KDA kernels use; the working set is either L2-resident (4 MB) or HBM-sized (1 GB).
//   PAT 1  16 B/lane, 64 lanes contiguous (1 KB per instruction)        "row-contiguous dwordx4"
//   PAT 2   8 B/lane contiguous (512 B)                                  "dwordx2"
//   PAT 3   4 B/lane contiguous (256 B)                                  "dword"
//   PAT 4  16 B/lane, 16 rows x 64 B (lane (m,q): row m, bytes 16q..)    s4b dq/dk/g f4 tiles
//   PAT 5   8 B/lane, 16 rows x 32 B                                     s4b k/q bf16 tiles, native frags
//   PAT 6   4 B/lane, 64 rows x 4 B (one row per lane)                   s4b akB/aqB transposed gathers
//   PAT 7  16 B/lane,  4 rows x 256 B                                    row-blocked 16 B tiles
//   PAT 8   8 B/lane,  4 rows x 128 B
// Row pitch 6144 B (= H*128 fp32 at H=12).  Reports wave-instructions per CU-cycle and bytes/s.
// hipcc -O3 --offload-arch=gfx936 ta_microbench.hip -o tamb && ./tamb
#include <hip/hip_runtime.h>
#include <cstdio>

constexpr int ROWB = 6144;

template <int PAT>
__device__ __forceinline__ size_t lane_off(int l) {
    const int m = l & 15, q = l >> 4;
    if (PAT == 1) return (size_t)l * 16;
    if (PAT == 2) return (size_t)l * 8;
    if (PAT == 3) return (size_t)l * 4;
    if (PAT == 4) return (size_t)m * ROWB + q * 16;
    if (PAT == 5) return (size_t)m * ROWB + q * 8;
    if (PAT == 7) return (size_t)(l >> 4) * ROWB + (l & 15) * 16;
    if (PAT == 8) return (size_t)(l >> 4) * ROWB + (l & 15) * 8;
    return (size_t)l * ROWB;
}

template <int PAT, int STORE>
__global__ void __launch_bounds__(256) k_ta(char* __restrict__ buf, int iters, size_t ws_mask, float* out) {
    const int tid = threadIdx.x, l = tid & 63, w = tid >> 6;
    const size_t lo = lane_off<PAT>(l);
    // each wave walks its own stride through the working set; 64-row footprint per instruction max
    const size_t span = 64 * (size_t)ROWB;
    size_t base = ((size_t)(blockIdx.x * 4 + w) * span * 3) & ws_mask;
    float acc = 0.f;
    for (int it = 0; it < iters; ++it) {
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            const size_t off = ((base + (size_t)(it * 8 + u) * span) & ws_mask) + lo;
            if (STORE) {
                if (PAT == 1 || PAT == 4 || PAT == 7) *reinterpret_cast<float4*>(buf + off) = make_float4(acc, 1.f, 2.f, 3.f);
                else if (PAT == 2 || PAT == 5 || PAT == 8) *reinterpret_cast<float2*>(buf + off) = make_float2(acc, 1.f);
                else *reinterpret_cast<float*>(buf + off) = acc;
                acc += 1.f;
            } else {
                if (PAT == 1 || PAT == 4 || PAT == 7) { const float4 v = *reinterpret_cast<const float4*>(buf + off); acc += (v.x + v.y) + (v.z + v.w); }
                else if (PAT == 2 || PAT == 5 || PAT == 8) { const float2 v = *reinterpret_cast<const float2*>(buf + off); acc += v.x + v.y; }
                else acc += *reinterpret_cast<const float*>(buf + off);
            }
        }
    }
    out[blockIdx.x * 256 + tid] = acc;
}

template <int PAT, int STORE>
static void run(char* buf, size_t ws, float* out, const char* name) {
    const int grid = 80 * 8, iters = 400;
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    hipLaunchKernelGGL((k_ta<PAT, STORE>), dim3(grid), dim3(256), 0, 0, buf, iters, ws - 1, out);
    hipDeviceSynchronize();
    float best = 1e30f;
    for (int r = 0; r < 5; ++r) {
        hipEventRecord(a);
        hipLaunchKernelGGL((k_ta<PAT, STORE>), dim3(grid), dim3(256), 0, 0, buf, iters, ws - 1, out);
        hipEventRecord(b); hipEventSynchronize(b);
        float ms; hipEventElapsedTime(&ms, a, b); best = ms < best ? ms : best;
    }
    const double insts = (double)grid * 4 * iters * 8;
    const int bpl = (PAT == 1 || PAT == 4 || PAT == 7) ? 16 : (PAT == 2 || PAT == 5 || PAT == 8) ? 8 : 4;
    const double cyc = best * 1e-3 * 1.5e9 * 80;               // CU-cycles
    printf("%-34s %s ws=%5zu MB: %6.3f wave-inst/CU-cycle (%5.1f cycles/inst/CU)  %7.0f GB/s\n", name,
           STORE ? "store" : "load ", ws >> 20, insts / cyc, cyc / insts, insts * 64 * bpl / (best * 1e-3) / 1e9);
}

int main() {
    char* buf; float* out;
    const size_t big = (size_t)1 << 30;
    hipMalloc(&buf, big + 64 * ROWB * 2); hipMalloc(&out, 80 * 8 * 256 * 4);
    hipMemset(buf, 0, big + 64 * ROWB * 2);
    for (size_t ws : {(size_t)4 << 20, big}) {
        run<1, 0>(buf, ws, out, "P1 16B/lane contiguous (1KB)");
        run<2, 0>(buf, ws, out, "P2  8B/lane contiguous (512B)");
        run<3, 0>(buf, ws, out, "P3  4B/lane contiguous (256B)");
        run<4, 0>(buf, ws, out, "P4 16B/lane 16 rows x 64B");
        run<5, 0>(buf, ws, out, "P5  8B/lane 16 rows x 32B");
        run<6, 0>(buf, ws, out, "P6  4B/lane 64 rows x 4B");
        run<7, 0>(buf, ws, out, "P7 16B/lane 4 rows x 256B");
        run<8, 0>(buf, ws, out, "P8  8B/lane 4 rows x 128B");
        run<1, 1>(buf, ws, out, "P1 16B/lane contiguous (1KB)");
        run<4, 1>(buf, ws, out, "P4 16B/lane 16 rows x 64B");
        run<5, 1>(buf, ws, out, "P5  8B/lane 16 rows x 32B");
        run<2, 1>(buf, ws, out, "P2  8B/lane contiguous (512B)");
        run<7, 1>(buf, ws, out, "P7 16B/lane 4 rows x 256B");
    }
    return 0;
}
