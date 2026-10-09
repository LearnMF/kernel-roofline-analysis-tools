// L0 machine calibration microbenchmarks (HIP, HCU/AMD wave64 targets).
//
// Every benchmark prints one JSON line per configuration:
//   {"bench": "...", "config": {...}, <measured fields>}
// kra/calibrate/run.py builds this file on the target, runs it and turns the lines
// into machine.json.  Peaks are "attainable" peaks: the best configuration found
// here, not datasheet numbers.  Every number is tied to the exact device/toolchain
// recorded by run.py.
//
// Build: hipcc -O3 -std=c++17 --offload-arch=<gfx> microbench.hip -o microbench
// Run:   ./microbench [bench ...]      (no args = all)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#define CK(x)                                                                            \
    do {                                                                                 \
        hipError_t e_ = (x);                                                             \
        if (e_ != hipSuccess) {                                                          \
            fprintf(stderr, "HIP error %s at %s:%d: %s\n", #x, __FILE__, __LINE__,       \
                    hipGetErrorString(e_));                                              \
            exit(1);                                                                     \
        }                                                                                \
    } while (0)

typedef short v4s __attribute__((ext_vector_type(4)));
typedef _Float16 v4h __attribute__((ext_vector_type(4)));
typedef float v2f __attribute__((ext_vector_type(2)));
typedef float v4f __attribute__((ext_vector_type(4)));
typedef int v4i __attribute__((ext_vector_type(4)));

static hipDeviceProp_t g_prop;
static int g_cus = 0;

// Times `fn` (which enqueues work on stream 0) with events: `warm` untimed runs, then
// `reps` individually timed runs.  Returns all samples in ms.
template <typename F>
static std::vector<float> time_ms(F&& fn, int warm = 2, int reps = 10) {
    hipEvent_t a, b;
    CK(hipEventCreate(&a));
    CK(hipEventCreate(&b));
    for (int i = 0; i < warm; ++i) fn();
    CK(hipDeviceSynchronize());
    std::vector<float> out;
    for (int i = 0; i < reps; ++i) {
        CK(hipEventRecord(a, 0));
        fn();
        CK(hipEventRecord(b, 0));
        CK(hipEventSynchronize(b));
        float ms = 0;
        CK(hipEventElapsedTime(&ms, a, b));
        out.push_back(ms);
    }
    CK(hipEventDestroy(a));
    CK(hipEventDestroy(b));
    return out;
}
static float vmin(std::vector<float> v) { return *std::min_element(v.begin(), v.end()); }
static float vmed(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// DTK defaults kernels to a 256-thread launch bound; benchmarks sweep up to 512 (1024
// for the barrier test), so every kernel declares its bound explicitly.
#define LB512 __launch_bounds__(512)
#define LB1024 __launch_bounds__(1024)

// ------------------------------------------------------------------ bandwidth
// U independent 16-byte loads per thread per iteration keep enough bytes in flight.
template <int U>
__global__ void LB512 k_read(const v4f* __restrict__ p, size_t n4, float* sink) {
    v4f acc = {0.f, 0.f, 0.f, 0.f};
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + (U - 1) * stride < n4; i += U * stride) {
#pragma unroll
        for (int u = 0; u < U; ++u) acc += p[i + u * stride];
    }
    for (; i < n4; i += stride) acc += p[i];
    const float s = acc.x + acc.y + acc.z + acc.w;
    if (s == 123456.789f) sink[0] = s;  // never true; defeats dead-code elimination
}

// Same as k_read<1> but with non-temporal (streaming) loads.
__global__ void LB512 k_read_nt(const v4f* __restrict__ p, size_t n4, float* sink) {
    v4f acc = {0.f, 0.f, 0.f, 0.f};
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride)
        acc += __builtin_nontemporal_load(p + i);
    const float s = acc.x + acc.y + acc.z + acc.w;
    if (s == 123456.789f) sink[0] = s;
}

template <int U>
__global__ void LB512 k_write(v4f* __restrict__ p, size_t n4, float v) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const v4f x = {v, v, v, v};
    for (; i + (U - 1) * stride < n4; i += U * stride) {
#pragma unroll
        for (int u = 0; u < U; ++u) p[i + u * stride] = x;
    }
    for (; i < n4; i += stride) p[i] = x;
}

template <int U>
__global__ void LB512 k_copy(const v4f* __restrict__ src, v4f* __restrict__ dst, size_t n4) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + (U - 1) * stride < n4; i += U * stride) {
        v4f t[U];
#pragma unroll
        for (int u = 0; u < U; ++u) t[u] = src[i + u * stride];
#pragma unroll
        for (int u = 0; u < U; ++u) dst[i + u * stride] = t[u];
    }
    for (; i < n4; i += stride) dst[i] = src[i];
}

// Cache-resident read: every thread issues exactly `loads` 16-byte loads (so the byte
// count is exact for any grid).  Load j of thread t reads element
// (t + j*stride + j*4099*64) & mask, so consecutive loads of one thread land on
// different lines and a CU does not keep re-reading the same L1-resident lines.
// n4 (working set / 16) is a power of two.
__global__ void LB512 k_read_resident(const v4f* __restrict__ p, size_t n4, int loads,
                                      float* sink) {
    v4f acc = {0.f, 0.f, 0.f, 0.f};
    const size_t mask = n4 - 1;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t step = stride + (size_t)4099 * 64;
    size_t idx = tid;
    for (int j = 0; j < loads; j += 4) {
        acc += p[idx & mask];
        acc += p[(idx + step) & mask];
        acc += p[(idx + 2 * step) & mask];
        acc += p[(idx + 3 * step) & mask];
        idx += 4 * step;
    }
    const float s = acc.x + acc.y + acc.z + acc.w;
    if (s == 123456.789f) sink[0] = s;
}

// LDS read bandwidth: conflict-free 16-byte reads, all CUs busy.
__global__ void __launch_bounds__(256) k_lds_read(int iters, float* sink) {
    __shared__ v4f s[1024];
    const int t = threadIdx.x;
    for (int k = t; k < 1024; k += 256) s[k] = v4f{(float)k, 1.f, 2.f, 3.f};
    __syncthreads();
    v4f a0 = {0, 0, 0, 0}, a1 = a0, a2 = a0, a3 = a0;
    int idx = t;
    for (int i = 0; i < iters; ++i) {
        a0 += s[idx];
        a1 += s[(idx + 256) & 1023];
        a2 += s[(idx + 512) & 1023];
        a3 += s[(idx + 768) & 1023];
        idx = (idx + 64) & 1023;
    }
    const v4f a = a0 + a1 + a2 + a3;
    const float r = a.x + a.y + a.z + a.w;
    if (r == 123456.789f) sink[0] = r;
}

// ------------------------------------------------------------------ compute peaks
// NCH independent accumulator chains per wave hide the instruction latency.
#define MMAC_BODY(BUILTIN, AT, BT, CT, INIT_A, INIT_B)                                   \
    AT a = INIT_A;                                                                       \
    BT b = INIT_B;                                                                       \
    CT c[NCH];                                                                           \
    _Pragma("unroll") for (int k = 0; k < NCH; ++k) c[k] = CT{};                         \
    for (int i = 0; i < iters; ++i) {                                                    \
        _Pragma("unroll") for (int k = 0; k < NCH; ++k) c[k] = BUILTIN(a, b, c[k]);      \
    }                                                                                    \
    CT acc = c[0];                                                                       \
    _Pragma("unroll") for (int k = 1; k < NCH; ++k) acc += c[k];                         \
    if (acc[0] == 123457) sink[0] = (float)acc[1];

template <int NCH>
__global__ void LB512 k_mmac_bf16(int iters, float* sink) {
    const short v = (short)(0x3f80 + (threadIdx.x & 7));
    MMAC_BODY(__builtin_amdgcn_mmac_f32_16x16x16bf16, v4s, v4s, v4f, (v4s{v, v, v, v}),
              (v4s{v, v, v, v}))
}
template <int NCH>
__global__ void LB512 k_mmac_f16(int iters, float* sink) {
    const _Float16 v = (_Float16)(1.0f + (threadIdx.x & 7) * 0.001f);
    MMAC_BODY(__builtin_amdgcn_mmac_f32_16x16x16f16, v4h, v4h, v4f, (v4h{v, v, v, v}),
              (v4h{v, v, v, v}))
}
template <int NCH>
__global__ void LB512 k_mmac_f32x8(int iters, float* sink) {
    const float v = 1.0f + (threadIdx.x & 7) * 0.001f;
    MMAC_BODY(__builtin_amdgcn_mmac_f32_16x16x8f32, v2f, v2f, v4f, (v2f{v, v}), (v2f{v, v}))
}
template <int NCH>
__global__ void LB512 k_mmac_tf32(int iters, float* sink) {
    const float v = 1.0f + (threadIdx.x & 7) * 0.001f;
    MMAC_BODY(__builtin_amdgcn_mmac_f32_16x16x8tf32, v2f, v2f, v4f, (v2f{v, v}), (v2f{v, v}))
}
template <int NCH>
__global__ void LB512 k_mmac_i8(int iters, float* sink) {
    const long v = 0x0101010101010101L + (threadIdx.x & 7);
    MMAC_BODY(__builtin_amdgcn_mmac_i32_16x16x32i8, long, long, v4i, v, v)
}

// FP32 FMA throughput (per-lane scalar FMAs; the compiler may emit v_fma_f32 or
// v_pk_fma_f32 -- run.py records which via the saved ISA).
template <int NCH>
__global__ void LB512 k_fma_f32(int iters, float* sink) {
    float x = 1.0f + threadIdx.x * 1e-7f, y = 0.999999f;
    float c[NCH];
#pragma unroll
    for (int k = 0; k < NCH; ++k) c[k] = (float)k * 1e-3f;
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int k = 0; k < NCH; ++k) c[k] = __builtin_fmaf(c[k], y, x);
    }
    float acc = 0;
#pragma unroll
    for (int k = 0; k < NCH; ++k) acc += c[k];
    if (acc == 123456.789f) sink[0] = acc;
}

// Transcendental (v_exp_f32) throughput.  exp2(-x) keeps values bounded in (0.5, 1].
template <int NCH>
__global__ void LB512 k_exp2(int iters, float* sink) {
    float c[NCH];
#pragma unroll
    for (int k = 0; k < NCH; ++k) c[k] = 0.5f + (threadIdx.x & 31) * 0.01f + k * 1e-3f;
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int k = 0; k < NCH; ++k) {
#if __has_builtin(__builtin_amdgcn_exp2f)
            c[k] = __builtin_amdgcn_exp2f(-c[k]);
#else
            c[k] = exp2f(-c[k]);
#endif
        }
    }
    float acc = 0;
#pragma unroll
    for (int k = 0; k < NCH; ++k) acc += c[k];
    if (acc == 123456.789f) sink[0] = acc;
}

// ------------------------------------------------------------------ latencies
// Single-lane pointer chase.  Records both clock64() (shader clock) and wall_clock64()
// (constant-rate clock) so run.py can convert cycles to ns and infer the shader clock.
__global__ void k_chase(const unsigned* __restrict__ p, int steps, unsigned start,
                        unsigned long long* out, unsigned* sink) {
    // A wave-uniform index would be compiled to s_load (scalar cache) -- pin it in a
    // VGPR so every step is a vector global_load through the vector memory path.
    unsigned j = start + threadIdx.x;  // launched with one lane, so == start
    for (int s = 0; s < 64; ++s) {     // warm TLB/cache for the resident cases
        asm volatile("" : "+v"(j));
        j = p[j];
    }
    const unsigned long long c0 = clock64(), w0 = wall_clock64();
    for (int s = 0; s < steps; ++s) {
        asm volatile("" : "+v"(j));
        j = p[j];
    }
    const unsigned long long c1 = clock64(), w1 = wall_clock64();
    out[0] = c1 - c0;
    out[1] = w1 - w0;
    sink[0] = j;
}

__global__ void k_lds_chase(int steps, unsigned long long* out, unsigned* sink) {
    __shared__ unsigned s[4096];
    for (int k = threadIdx.x; k < 4096; k += blockDim.x) s[k] = (k * 1031 + 17) & 4095;
    __syncthreads();
    if (threadIdx.x != 0) return;
    unsigned j = 0;
    const unsigned long long c0 = clock64(), w0 = wall_clock64();
    for (int i = 0; i < steps; ++i) j = s[j];
    const unsigned long long c1 = clock64(), w1 = wall_clock64();
    out[0] = c1 - c0;
    out[1] = w1 - w0;
    sink[0] = j;
}

// Dependent MMAC chain in one wave: issue-to-result latency of the matrix pipe.
__global__ void k_mmac_latency(int iters, unsigned long long* out, float* sink) {
    v4s a = {0x3f80, 0x3f80, 0x3f80, 0x3f80};
    v4f c = {0, 0, 0, 0};
    const unsigned long long c0 = clock64(), w0 = wall_clock64();
    for (int i = 0; i < iters; ++i) c = __builtin_amdgcn_mmac_f32_16x16x16bf16(a, a, c);
    const unsigned long long c1 = clock64(), w1 = wall_clock64();
    if (threadIdx.x == 0) {
        out[0] = c1 - c0;
        out[1] = w1 - w0;
    }
    if (c[0] == 123456.f) sink[0] = c[1];
}

// CTA barrier cost.  MODE 0: __syncthreads(); MODE 1: LDS-only barrier
// (s_waitcnt lgkmcnt(0); s_barrier), the form used by hand-written kernels.
template <int MODE>
__global__ void LB1024 k_barrier(int iters, unsigned long long* out) {
    __shared__ int s[1024];
    s[threadIdx.x] = threadIdx.x;
    __syncthreads();
    const unsigned long long c0 = clock64(), w0 = wall_clock64();
    for (int i = 0; i < iters; ++i) {
        if (MODE == 0) {
            __syncthreads();
        } else {
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
            __builtin_amdgcn_sched_barrier(0);
        }
    }
    const unsigned long long c1 = clock64(), w1 = wall_clock64();
    if (threadIdx.x == 0) {
        out[0] = c1 - c0;
        out[1] = w1 - w0;
    }
}

__global__ void k_empty() {}

// ------------------------------------------------------------------ host drivers
static float* g_sink = nullptr;

static int grid_for(int per_cu) { return g_cus * per_cu; }

template <int U>
static void bw_case(const char* kind, v4f* a, v4f* b, size_t n4, int per_cu, int block) {
    const int grid = grid_for(per_cu);
    std::vector<float> t;
    double bytes = 0;
    if (!strcmp(kind, "read")) {
        t = time_ms([&] { hipLaunchKernelGGL(k_read<U>, grid, block, 0, 0, a, n4, g_sink); });
        bytes = (double)n4 * 16;
    } else if (!strcmp(kind, "read_nt")) {
        if (U != 1) return;
        t = time_ms([&] { hipLaunchKernelGGL(k_read_nt, grid, block, 0, 0, a, n4, g_sink); });
        bytes = (double)n4 * 16;
    } else if (!strcmp(kind, "write")) {
        t = time_ms([&] { hipLaunchKernelGGL(k_write<U>, grid, block, 0, 0, a, n4, 1.f); });
        bytes = (double)n4 * 16;
    } else {
        t = time_ms([&] { hipLaunchKernelGGL(k_copy<U>, grid, block, 0, 0, a, b, n4); });
        bytes = (double)n4 * 32;
    }
    CK(hipGetLastError());
    printf("{\"bench\":\"hbm_%s\",\"config\":{\"bytes_per_pass\":%.0f,\"grid\":%d,\"block\":%d,"
           "\"unroll\":%d},\"ms_min\":%.5f,\"ms_med\":%.5f,\"GBps_best\":%.1f,\"GBps_med\":%.1f}\n",
           kind, bytes, grid, block, U, vmin(t), vmed(t), bytes / vmin(t) / 1e6,
           bytes / vmed(t) / 1e6);
    fflush(stdout);
}

static void bench_hbm() {
    const size_t bytes = (size_t)1 << 30;  // 1 GiB per buffer, far beyond L2
    const size_t n4 = bytes / 16;
    v4f *a, *b;
    CK(hipMalloc(&a, bytes));
    CK(hipMalloc(&b, bytes));
    CK(hipMemset(a, 0, bytes));
    CK(hipMemset(b, 0, bytes));
    for (const char* kind : {"read", "read_nt", "write", "copy"})
        for (int per_cu : {4, 8, 16, 32})
            for (int block : {256, 512}) {
                bw_case<1>(kind, a, b, n4, per_cu, block);
                bw_case<4>(kind, a, b, n4, per_cu, block);
            }
    CK(hipFree(a));
    CK(hipFree(b));
}

static void bench_l2() {
    const size_t l2 = g_prop.l2CacheSize > 0 ? (size_t)g_prop.l2CacheSize : (4u << 20);
    v4f* a;
    CK(hipMalloc(&a, l2));
    CK(hipMemset(a, 0, l2));
    // Power-of-two working sets: two inside L2 (resident) and one 4x larger (spills to
    // HBM) as a sanity check that the resident numbers really come from L2.
    size_t ws = 1;
    while (ws * 2 <= l2 / 2) ws *= 2;
    const size_t big = ws * 8;
    CK(hipFree(a));
    CK(hipMalloc(&a, big));
    CK(hipMemset(a, 0, big));
    for (size_t w : {ws / 2, ws, big}) {
        const size_t n4 = w / 16;
        const int loads = 1024;
        for (int per_cu : {4, 8, 16}) {
            for (int block : {256, 512}) {
                const int grid = grid_for(per_cu);
                auto t = time_ms([&] {
                    hipLaunchKernelGGL(k_read_resident, grid, block, 0, 0, a, n4, loads, g_sink);
                });
                CK(hipGetLastError());
                const double total = (double)grid * block * loads * 16;
                printf("{\"bench\":\"l2_read\",\"config\":{\"working_set\":%zu,\"l2_size\":%zu,"
                       "\"resident\":%s,\"grid\":%d,\"block\":%d,\"loads_per_thread\":%d},"
                       "\"ms_min\":%.5f,\"GBps_best\":%.1f,\"GBps_med\":%.1f}\n",
                       w, l2, w <= l2 / 2 ? "true" : "false", grid, block, loads, vmin(t),
                       total / vmin(t) / 1e6, total / vmed(t) / 1e6);
                fflush(stdout);
            }
        }
    }
    CK(hipFree(a));
}

static void bench_lds() {
    const int iters = 4096, block = 256;
    for (int per_cu : {4, 8, 16}) {
        const int grid = grid_for(per_cu);
        auto t = time_ms([&] { hipLaunchKernelGGL(k_lds_read, grid, block, 0, 0, iters, g_sink); });
        CK(hipGetLastError());
        const double bytes = (double)grid * block * iters * 4 * 16;
        printf("{\"bench\":\"lds_read\",\"config\":{\"grid\":%d,\"block\":%d,\"iters\":%d},"
               "\"ms_min\":%.5f,\"GBps_best\":%.1f,\"GBps_med\":%.1f}\n",
               grid, block, iters, vmin(t), bytes / vmin(t) / 1e6, bytes / vmed(t) / 1e6);
        fflush(stdout);
    }
}

// ops_per_inst_wave = arithmetic ops (2 per FMA) performed by ONE wave-level instruction.
template <typename K>
static void compute_case(const char* name, K kernel, int nch, double ops_per_inst_wave,
                         int per_cu, int block, int iters) {
    const int grid = grid_for(per_cu);
    auto t = time_ms([&] { hipLaunchKernelGGL(kernel, grid, block, 0, 0, iters, g_sink); });
    CK(hipGetLastError());
    const double waves = (double)grid * (block / g_prop.warpSize);
    const double ops = waves * iters * nch * ops_per_inst_wave;
    printf("{\"bench\":\"%s\",\"config\":{\"grid\":%d,\"block\":%d,\"chains\":%d,\"iters\":%d,"
           "\"ops_per_inst_wave\":%.0f},\"ms_min\":%.5f,\"TOPS_best\":%.3f,\"TOPS_med\":%.3f}\n",
           name, grid, block, nch, iters, ops_per_inst_wave, vmin(t), ops / vmin(t) / 1e9,
           ops / vmed(t) / 1e9);
    fflush(stdout);
}

static void bench_compute() {
    const double mm16 = 2.0 * 16 * 16 * 16, mm8 = 2.0 * 16 * 16 * 8, mmi8 = 2.0 * 16 * 16 * 32;
    // iters sized so each launch runs several ms (launch/tail effects < 1%).
    const double vw = g_prop.warpSize;
    for (int per_cu : {4, 8}) {
        for (int block : {256, 512}) {
            compute_case("mmac_bf16", k_mmac_bf16<4>, 4, mm16, per_cu, block, 32768);
            compute_case("mmac_bf16", k_mmac_bf16<8>, 8, mm16, per_cu, block, 16384);
            compute_case("mmac_f16", k_mmac_f16<8>, 8, mm16, per_cu, block, 16384);
            compute_case("mmac_f32x8", k_mmac_f32x8<8>, 8, mm8, per_cu, block, 4096);
            compute_case("mmac_tf32", k_mmac_tf32<8>, 8, mm8, per_cu, block, 16384);
            compute_case("mmac_i8", k_mmac_i8<8>, 8, mmi8, per_cu, block, 16384);
            compute_case("valu_fma_f32", k_fma_f32<8>, 8, 2.0 * vw, per_cu, block, 65536);
            compute_case("valu_fma_f32", k_fma_f32<16>, 16, 2.0 * vw, per_cu, block, 32768);
            compute_case("sfu_exp2", k_exp2<8>, 8, 1.0 * vw, per_cu, block, 16384);
        }
    }
}

static void latency_line(const char* name, const char* cfg, unsigned long long* d_out,
                         double steps) {
    unsigned long long h[2];
    CK(hipMemcpy(h, d_out, sizeof(h), hipMemcpyDeviceToHost));
    int wall_khz = 0;
    CK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, 0));
    const double ns = (double)h[1] / (wall_khz * 1e3) * 1e9;
    printf("{\"bench\":\"%s\",\"config\":{%s},\"cycles_per_step\":%.2f,\"ns_per_step\":%.2f,"
           "\"shader_clock_mhz\":%.1f,\"wall_clock_khz\":%d}\n",
           name, cfg, h[0] / steps, ns / steps, h[1] ? h[0] / (double)h[1] * wall_khz / 1e3 : 0.0,
           wall_khz);
    fflush(stdout);
}

static void bench_latency() {
    unsigned long long* d_out;
    unsigned* d_sink;
    CK(hipMalloc(&d_out, 16));
    CK(hipMalloc(&d_sink, 4));
    // Global pointer chase over a random cycle; one element per 256 B so every step
    // touches a new cache line.
    for (size_t ws : {(size_t)16 << 10, (size_t)2 << 20, (size_t)512 << 20}) {
        const size_t line = 64;  // unsigned elements per 256 B
        const size_t nodes = ws / 256;
        std::vector<unsigned> order(nodes);
        std::iota(order.begin(), order.end(), 0u);
        std::shuffle(order.begin() + 1, order.end(), std::mt19937(1234));
        std::vector<unsigned> h(ws / 4, 0);
        for (size_t k = 0; k < nodes; ++k)
            h[order[k] * line] = (unsigned)(order[(k + 1) % nodes] * line);
        unsigned* d;
        CK(hipMalloc(&d, ws));
        CK(hipMemcpy(d, h.data(), ws, hipMemcpyHostToDevice));
        const int steps = 20000;
        hipLaunchKernelGGL(k_chase, 1, 1, 0, 0, d, steps, 0u, d_out, d_sink);
        CK(hipDeviceSynchronize());
        hipLaunchKernelGGL(k_chase, 1, 1, 0, 0, d, steps, 0u, d_out, d_sink);
        CK(hipDeviceSynchronize());
        char cfg[96];
        snprintf(cfg, sizeof(cfg), "\"working_set\":%zu,\"steps\":%d", ws, steps);
        latency_line("gmem_latency", cfg, d_out, steps);
        CK(hipFree(d));
    }
    {
        const int steps = 100000;
        hipLaunchKernelGGL(k_lds_chase, 1, 64, 0, 0, steps, d_out, d_sink);
        CK(hipDeviceSynchronize());
        latency_line("lds_latency", "\"steps\":100000", d_out, steps);
    }
    {
        const int iters = 100000;
        hipLaunchKernelGGL(k_mmac_latency, 1, 64, 0, 0, iters, d_out, g_sink);
        CK(hipDeviceSynchronize());
        latency_line("mmac_bf16_latency", "\"iters\":100000", d_out, iters);
    }
    for (int block : {256, 512, 1024}) {
        const int iters = 100000;
        char cfg[64];
        hipLaunchKernelGGL(k_barrier<0>, 1, block, 0, 0, iters, d_out);
        CK(hipDeviceSynchronize());
        snprintf(cfg, sizeof(cfg), "\"block\":%d,\"mode\":\"syncthreads\"", block);
        latency_line("barrier", cfg, d_out, iters);
        hipLaunchKernelGGL(k_barrier<1>, 1, block, 0, 0, iters, d_out);
        CK(hipDeviceSynchronize());
        snprintf(cfg, sizeof(cfg), "\"block\":%d,\"mode\":\"lds_barrier\"", block);
        latency_line("barrier", cfg, d_out, iters);
    }
    CK(hipFree(d_out));
    CK(hipFree(d_sink));
}

static void bench_launch() {
    const int n = 2000;
    // Back-to-back empty kernels on one stream: steady-state per-launch cost
    // (max of host enqueue rate and device dispatch rate).
    auto t = time_ms([&] {
        for (int i = 0; i < n; ++i) hipLaunchKernelGGL(k_empty, 1, 64, 0, 0);
    }, 1, 5);
    printf("{\"bench\":\"launch_stream\",\"config\":{\"kernels\":%d},\"us_per_kernel_best\":%.3f,"
           "\"us_per_kernel_med\":%.3f}\n", n, vmin(t) * 1e3 / n, vmed(t) * 1e3 / n);
    // Same work replayed from a HIP graph: what graph capture can recover.
    hipStream_t s;
    CK(hipStreamCreate(&s));
    hipGraph_t g;
    hipGraphExec_t ge;
    CK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
    for (int i = 0; i < n; ++i) hipLaunchKernelGGL(k_empty, 1, 64, 0, s);
    CK(hipStreamEndCapture(s, &g));
    CK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    hipEvent_t a, b;
    CK(hipEventCreate(&a));
    CK(hipEventCreate(&b));
    std::vector<float> tg;
    for (int r = 0; r < 6; ++r) {
        CK(hipEventRecord(a, s));
        CK(hipGraphLaunch(ge, s));
        CK(hipEventRecord(b, s));
        CK(hipEventSynchronize(b));
        float ms;
        CK(hipEventElapsedTime(&ms, a, b));
        if (r) tg.push_back(ms);
    }
    printf("{\"bench\":\"launch_graph\",\"config\":{\"kernels\":%d},\"us_per_kernel_best\":%.3f,"
           "\"us_per_kernel_med\":%.3f}\n", n, vmin(tg) * 1e3 / n, vmed(tg) * 1e3 / n);
    CK(hipGraphExecDestroy(ge));
    CK(hipGraphDestroy(g));
    CK(hipStreamDestroy(s));
    // Launch + synchronize round trip of a single kernel.
    std::vector<float> rt;
    for (int r = 0; r < 200; ++r) {
        CK(hipEventRecord(a, 0));
        hipLaunchKernelGGL(k_empty, 1, 64, 0, 0);
        CK(hipEventRecord(b, 0));
        CK(hipEventSynchronize(b));
        float ms;
        CK(hipEventElapsedTime(&ms, a, b));
        if (r >= 10) rt.push_back(ms);
    }
    printf("{\"bench\":\"launch_single\",\"config\":{\"samples\":%zu},\"us_best\":%.3f,"
           "\"us_med\":%.3f}\n", rt.size(), vmin(rt) * 1e3, vmed(rt) * 1e3);
    CK(hipEventDestroy(a));
    CK(hipEventDestroy(b));
    fflush(stdout);
}

static void bench_pcie() {
    const size_t bytes = (size_t)256 << 20;
    void *h, *d;
    CK(hipHostMalloc(&h, bytes));
    CK(hipMalloc(&d, bytes));
    memset(h, 1, bytes);
    auto th = time_ms([&] { CK(hipMemcpyAsync(d, h, bytes, hipMemcpyHostToDevice, 0)); }, 1, 5);
    auto td = time_ms([&] { CK(hipMemcpyAsync(h, d, bytes, hipMemcpyDeviceToHost, 0)); }, 1, 5);
    printf("{\"bench\":\"pcie_h2d\",\"config\":{\"bytes\":%zu,\"pinned\":true},\"GBps_best\":%.2f}\n",
           bytes, bytes / vmin(th) / 1e6);
    printf("{\"bench\":\"pcie_d2h\",\"config\":{\"bytes\":%zu,\"pinned\":true},\"GBps_best\":%.2f}\n",
           bytes, bytes / vmin(td) / 1e6);
    CK(hipHostFree(h));
    CK(hipFree(d));
    fflush(stdout);
}

static void print_device() {
    int wall_khz = 0;
    CK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, 0));
    printf("{\"bench\":\"device\",\"name\":\"%s\",\"gcnArchName\":\"%s\",\"cu_count\":%d,"
           "\"warp_size\":%d,\"clock_khz\":%d,\"memory_clock_khz\":%d,\"memory_bus_width\":%d,"
           "\"l2_bytes\":%d,\"lds_per_block\":%zu,\"lds_per_cu\":%zu,\"regs_per_block\":%d,"
           "\"max_threads_per_cu\":%d,\"global_mem_bytes\":%zu,\"wall_clock_khz\":%d}\n",
           g_prop.name, g_prop.gcnArchName, g_prop.multiProcessorCount, g_prop.warpSize,
           g_prop.clockRate, g_prop.memoryClockRate, g_prop.memoryBusWidth, g_prop.l2CacheSize,
           g_prop.sharedMemPerBlock, g_prop.maxSharedMemoryPerMultiProcessor, g_prop.regsPerBlock,
           g_prop.maxThreadsPerMultiProcessor, g_prop.totalGlobalMem, wall_khz);
    fflush(stdout);
}

int main(int argc, char** argv) {
    CK(hipGetDeviceProperties(&g_prop, 0));
    g_cus = g_prop.multiProcessorCount;
    CK(hipMalloc(&g_sink, 64));
    print_device();
    auto want = [&](const char* b) {
        if (argc < 2) return true;
        for (int i = 1; i < argc; ++i)
            if (!strcmp(argv[i], b)) return true;
        return false;
    };
    if (want("hbm")) bench_hbm();
    if (want("l2")) bench_l2();
    if (want("lds")) bench_lds();
    if (want("compute")) bench_compute();
    if (want("latency")) bench_latency();
    if (want("launch")) bench_launch();
    if (want("pcie")) bench_pcie();
    CK(hipFree(g_sink));
    return 0;
}
