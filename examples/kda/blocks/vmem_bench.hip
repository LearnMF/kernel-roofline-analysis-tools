// Building-block library P0-2: VMEM cost per wave instruction on gfx936 (docs/tasks/kda_blocks_task.md).
// One wave instruction = 64 lanes x BPL bytes; lane l -> row l / LPR (LPR = 64 / rows), column
// ((l % LPR) / DUP) * BPL inside the row; rows are PITCH bytes apart ([T, H, X] row-major: pitch = H * X * dtype).
// Per instruction:
//   mode 0 (scatter): consecutive instructions of a wave start far apart (span), like fragment tiles of different
//                     row blocks / chunks
//   mode 1 (stream) : consecutive instructions continue at the next row group (base += rows * pitch), like
//                     bwd_tail walking down its 16 rows
// Throughput: 32 waves / CU, 8 independent instructions per loop iteration -> CU cycles per wave instruction.
// Latency   : 1 wave / CU, each load's address depends on the previous load (pointer chase through asm).
//   hipcc -O3 --offload-arch=gfx936 vmem_bench.hip -o vmemb && ./vmemb > vmem_bench.txt
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>

template <int B> struct V;
template <> struct V<2> { typedef unsigned short T; };
template <> struct V<4> { typedef float T; };
template <> struct V<8> { typedef float2 T; };
template <> struct V<16> { typedef float4 T; };

__device__ __forceinline__ float sum(unsigned short v) { return (float)v; }
__device__ __forceinline__ float sum(float v) { return v; }
__device__ __forceinline__ float sum(float2 v) { return v.x + v.y; }
__device__ __forceinline__ float sum(float4 v) { return (v.x + v.y) + (v.z + v.w); }
__device__ __forceinline__ void mk(unsigned short& v, float a) { v = (unsigned short)a; }
__device__ __forceinline__ void mk(float& v, float a) { v = a; }
__device__ __forceinline__ void mk(float2& v, float a) { v = make_float2(a, 1.f); }
__device__ __forceinline__ void mk(float4& v, float a) { v = make_float4(a, 1.f, 2.f, 3.f); }

struct P {
    char* buf;
    size_t mask;      // working set - 1 (power of two)
    size_t pitch, span;
    int iters, lpr, dup, mode;
    float* out;
};

template <int BPL, int ST>
__global__ void __launch_bounds__(256) k_thr(P p) {
    typedef typename V<BPL>::T T;
    const int tid = threadIdx.x, l = tid & 63;
    const int wave = blockIdx.x * 4 + (tid >> 6);
    const size_t loff = (size_t)(l / p.lpr) * p.pitch + (size_t)(((l % p.lpr) / p.dup) * BPL);
    const size_t rows = 64 / p.lpr;
    const size_t step = p.mode ? rows * p.pitch : p.span;
    size_t base = ((size_t)wave * 7919u * p.span) & p.mask & ~(size_t)255;
    float acc = 0.f;
    for (int it = 0; it < p.iters; ++it) {
        size_t o[8];
        #pragma unroll
        for (int u = 0; u < 8; ++u) o[u] = ((base + (size_t)(it * 8 + u) * step) & p.mask) + loff;
        if (ST) {
            #pragma unroll
            for (int u = 0; u < 8; ++u) { T v; mk(v, acc + u); *reinterpret_cast<T*>(p.buf + o[u]) = v; }
            acc += 1.f;
        } else {
            T v[8];
            #pragma unroll
            for (int u = 0; u < 8; ++u) v[u] = *reinterpret_cast<const T*>(p.buf + o[u]);
            #pragma unroll
            for (int u = 0; u < 8; ++u) acc += sum(v[u]);
        }
    }
    if (acc == 1234567.f) p.out[blockIdx.x * 256 + tid] = acc;   // keep the loads live, never true
}

template <int BPL>
__global__ void __launch_bounds__(64) k_lat(P p) {
    typedef typename V<BPL>::T T;
    const int l = threadIdx.x;
    const size_t loff = (size_t)(l / p.lpr) * p.pitch + (size_t)(((l % p.lpr) / p.dup) * BPL);
    size_t base = ((size_t)blockIdx.x * 104729u * p.span) & p.mask & ~(size_t)255;
    float acc = 0.f;
    int dep = 0;
    for (int it = 0; it < p.iters; ++it) {
        const size_t o = ((base + (size_t)it * p.span) & p.mask) + loff + (size_t)dep;
        const T v = *reinterpret_cast<const T*>(p.buf + o);
        const float s = sum(v);
        acc += s;
        int z = __float_as_int(s) & 0;           // buffer is zero: z == 0, but the compiler cannot know
        asm volatile("" : "+v"(z));
        dep = z;
    }
    if (acc == 1234567.f) p.out[blockIdx.x * 64 + l] = acc;
}

static float time_ms(void (*launch)(const P&), const P& p, int reps = 5) {
    hipEvent_t a, b;
    hipEventCreate(&a); hipEventCreate(&b);
    launch(p);
    hipDeviceSynchronize();
    float best = 1e30f;
    for (int r = 0; r < reps; ++r) {
        hipEventRecord(a);
        launch(p);
        hipEventRecord(b);
        hipEventSynchronize(b);
        float ms;
        hipEventElapsedTime(&ms, a, b);
        best = ms < best ? ms : best;
    }
    return best;
}

constexpr int GRID = 80 * 8;
template <int BPL, int ST> void L_thr(const P& p) { hipLaunchKernelGGL((k_thr<BPL, ST>), dim3(GRID), dim3(256), 0, 0, p); }
template <int BPL> void L_lat(const P& p) { hipLaunchKernelGGL((k_lat<BPL>), dim3(80), dim3(64), 0, 0, p); }

template <int BPL>
static void sweep(char* buf, float* out, size_t ws, const char* src, double clk_ghz) {
    const int rows_l[] = {1, 2, 4, 8, 16, 32, 64};
    const size_t pitches[] = {3072, 6144, 12288, 24576, 49152};
    for (int rows : rows_l) {
        for (int mode = 0; mode < 2; ++mode) {
            for (size_t pitch : pitches) {
                if (rows == 1 && mode == 0 && pitch != 3072) continue;   // pitch irrelevant
                P p{buf, ws - 1, pitch, 0, 0, 64 / rows, 1, mode, out};
                p.span = ((size_t)rows * pitch + 4095) & ~(size_t)4095;
                if (p.span < 4096) p.span = 4096;
                p.iters = 200;
                const double insts = (double)GRID * 4 * p.iters * 8;
                const float ld = time_ms(L_thr<BPL, 0>, p), st = time_ms(L_thr<BPL, 1>, p);
                const double cyc_ld = ld * 1e-3 * clk_ghz * 1e9 * 80 / insts, cyc_st = st * 1e-3 * clk_ghz * 1e9 * 80 / insts;
                float lat = -1.f;
                if (mode == 0 && (pitch == 12288 || rows == 1)) {
                    P q = p;
                    q.iters = 2000;
                    lat = time_ms(L_lat<BPL>, q, 3) * 1e-3 * clk_ghz * 1e9 / q.iters;
                }
                printf("VMEM src=%s bpl=%d rows=%d row_bytes=%d dup=1 pitch=%zu mode=%s ld_cyc=%.2f st_cyc=%.2f ld_GBs=%.0f st_GBs=%.0f lat_cyc=%.0f\n",
                       src, BPL, rows, 64 / rows * BPL, pitch, mode ? "stream" : "scatter", cyc_ld, cyc_st,
                       insts * 64 * BPL / (ld * 1e-3) / 1e9, insts * 64 * BPL / (st * 1e-3) / 1e9, lat);
                fflush(stdout);
            }
        }
    }
}

static void special(char* buf, float* out, size_t ws, const char* src, double clk_ghz) {
    // rowvec16: 16 rows, the 4 lanes of a row read the same 4 B (rstd / chunk-end g row); uniform: whole wave, one address
    struct S { const char* name; int lpr, dup; } sp[] = {{"rowvec16", 4, 4}, {"uniform", 64, 64}};
    for (auto s : sp) {
        for (size_t pitch : {(size_t)3072, (size_t)12288, (size_t)49152}) {
            P p{buf, ws - 1, pitch, 0, 200, s.lpr, s.dup, 0, out};
            p.span = ((size_t)(64 / s.lpr) * pitch + 4095) & ~(size_t)4095;
            const double insts = (double)GRID * 4 * p.iters * 8;
            const float ld = time_ms(L_thr<4, 0>, p), st = time_ms(L_thr<4, 1>, p);
            printf("VMEM src=%s bpl=4 rows=%d row_bytes=4 dup=%d pitch=%zu mode=%s ld_cyc=%.2f st_cyc=%.2f ld_GBs=0 st_GBs=0 lat_cyc=-1\n",
                   src, 64 / s.lpr, s.dup, pitch, s.name, ld * 1e-3 * clk_ghz * 1e9 * 80 / insts,
                   st * 1e-3 * clk_ghz * 1e9 * 80 / insts);
            fflush(stdout);
        }
    }
}

int main(int argc, char** argv) {
    const double clk = argc > 1 ? atof(argv[1]) : 1.5;
    const size_t big = (size_t)1 << 30, extra = (size_t)64 * 49152 + (1 << 20);
    char* buf;
    float* out;
    hipMalloc(&buf, big + extra);
    hipMalloc(&out, GRID * 256 * 4);
    hipMemset(buf, 0, big + extra);
    printf("# clk_ghz=%.3f grid=%d (32 waves/CU) iters=200x8 insts/wave\n", clk, GRID);
    for (int s = 0; s < 2; ++s) {
        const size_t ws = s ? big : ((size_t)4 << 20);
        const char* src = s ? "hbm" : "l2";
        sweep<2>(buf, out, ws, src, clk);
        sweep<4>(buf, out, ws, src, clk);
        sweep<8>(buf, out, ws, src, clk);
        sweep<16>(buf, out, ws, src, clk);
        special(buf, out, ws, src, clk);
    }
    return 0;
}
