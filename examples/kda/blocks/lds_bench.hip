// Building-block library P0-3: LDS instructions, the LDS exchange step and s_barrier on gfx936.
//   OP   : one LDS instruction form (inline asm, exact encoding), lane address = lane * stride (bytes); the bank
//          conflict degree is computed on the host (32 banks x 4 B, per 32-lane half, identical addresses broadcast).
//          Throughput: 16 waves / CU, 8 independent ops per iteration.  Latency: 1 wave / CU, dependent chain.
//   XCHG : W waves write X bytes each (ds_write_b128), s_barrier, read X bytes of another wave (ds_read_b128),
//          s_barrier -- the unit of every recurrence step and in-kernel transpose; CTAs / CU = 1..3.
//   BAR  : s_barrier alone (W = 4 / 8 / 16 waves), and with wave 0 arriving K dependent VALU ops late.
//   hipcc -O3 --offload-arch=gfx936 lds_bench.hip -o ldsb && ./ldsb > lds_bench.txt
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

typedef unsigned u2 __attribute__((ext_vector_type(2)));
typedef unsigned u4 __attribute__((ext_vector_type(4)));

enum Op { RB32, RB64, RB128, R2B32, RB16, WB32, WB64, WB128, BPERM };
static const char* NAME[] = {"ds_read_b32", "ds_read_b64", "ds_read_b128", "ds_read2_b32", "ds_read_u16",
                             "ds_write_b32", "ds_write_b64", "ds_write_b128", "ds_bpermute_b32"};
static const int BYTES[] = {4, 8, 16, 8, 2, 4, 8, 16, 4};

template <int OP>
__device__ __forceinline__ unsigned lds_op(unsigned a, unsigned x) {
    unsigned r = 0;
    if (OP == RB32) { asm volatile("ds_read_b32 %0, %1" : "=v"(r) : "v"(a)); }
    if (OP == RB64) { u2 v; asm volatile("ds_read_b64 %0, %1" : "=v"(v) : "v"(a)); r = v.x ^ v.y; }
    if (OP == RB128) { u4 v; asm volatile("ds_read_b128 %0, %1" : "=v"(v) : "v"(a)); r = v.x ^ v.w; }
    if (OP == R2B32) { u2 v; asm volatile("ds_read2_b32 %0, %1 offset1:32" : "=v"(v) : "v"(a)); r = v.x ^ v.y; }
    if (OP == RB16) { asm volatile("ds_read_u16 %0, %1" : "=v"(r) : "v"(a)); }
    if (OP == WB32) { asm volatile("ds_write_b32 %0, %1" :: "v"(a), "v"(x)); }
    if (OP == WB64) { u2 v = {x, x}; asm volatile("ds_write_b64 %0, %1" :: "v"(a), "v"(v)); }
    if (OP == WB128) { u4 v = {x, x, x, x}; asm volatile("ds_write_b128 %0, %1" :: "v"(a), "v"(v)); }
    if (OP == BPERM) { asm volatile("ds_bpermute_b32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(x)); }
    return r;
}

template <int OP>
__global__ void __launch_bounds__(256) k_op(int iters, int stride, unsigned* out) {
    __shared__ __align__(16) unsigned char s[32768];
    const int l = threadIdx.x & 63;
    const unsigned a = (unsigned)((l * stride) & (16383 & ~15)) + (unsigned)(threadIdx.x >> 6) * 16384u * 0;
    const unsigned base = (unsigned)(uintptr_t)s;
    unsigned acc = threadIdx.x;
    for (int it = 0; it < iters; ++it) {
        #pragma unroll
        for (int u = 0; u < 8; ++u) acc ^= lds_op<OP>(base + a + (OP == BPERM ? 0u : (unsigned)(u & 1) * 16384u), acc + u);
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    }
    if (acc == 0x9e3779b9u) out[blockIdx.x * 256 + threadIdx.x] = acc;
}

__global__ void __launch_bounds__(64) k_lat(int iters, unsigned* out) {
    __shared__ __align__(16) unsigned s[1024];
    const int l = threadIdx.x;
    for (int i = l; i < 1024; i += 64) s[i] = (unsigned)(uintptr_t)(s + ((i + 64) & 1023));   // next = +64 words
    __syncthreads();
    unsigned a = (unsigned)(uintptr_t)(s + l);
    for (int it = 0; it < iters; ++it) asm volatile("ds_read_b32 %0, %0\n s_waitcnt lgkmcnt(0)" : "+v"(a));
    if (a == 1u) out[blockIdx.x] = a;
}

// exchange step: write X bytes per wave -> barrier -> read another wave's X bytes -> barrier
template <int W>
__global__ void __launch_bounds__(64 * W) k_xchg(int iters, int xb, unsigned* out) {
    extern __shared__ __align__(16) unsigned char s[];
    const int tid = threadIdx.x, w = tid >> 6, l = tid & 63;
    const unsigned base = (unsigned)(uintptr_t)s;
    const int n = xb / 1024;                                      // b128 instructions per wave
    const unsigned mine = base + (unsigned)(w * xb + l * 16), other = base + (unsigned)(((w + 1) % W) * xb + l * 16);
    unsigned acc = tid;
    for (int it = 0; it < iters; ++it) {
        for (int j = 0; j < n; ++j) lds_op<WB128>(mine + (unsigned)j * 1024u, acc);
        asm volatile("s_waitcnt lgkmcnt(0)\n s_barrier" ::: "memory");
        for (int j = 0; j < n; ++j) acc ^= lds_op<RB128>(other + (unsigned)j * 1024u, 0);
        asm volatile("s_waitcnt lgkmcnt(0)\n s_barrier" ::: "memory");
    }
    if (acc == 0x9e3779b9u) out[blockIdx.x * 64 * W + tid] = acc;
}

template <int W>
__global__ void __launch_bounds__(64 * W) k_bar(int iters, int skew, unsigned* out) {
    const int w = threadIdx.x >> 6;
    float x = threadIdx.x;
    for (int it = 0; it < iters; ++it) {
        if (w == 0) for (int k = 0; k < skew; ++k) asm volatile("v_fma_f32 %0, %0, %0, 1.0" : "+v"(x));
        asm volatile("s_barrier" ::: "memory");
    }
    if (x == 1.2345f) out[threadIdx.x] = 0;
}

static double clk = 1.5e9;

template <class F>
static float best_ms(F launch, int reps = 5) {
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    launch(); hipDeviceSynchronize();
    float best = 1e30f;
    for (int r = 0; r < reps; ++r) {
        hipEventRecord(a); launch(); hipEventRecord(b); hipEventSynchronize(b);
        float ms; hipEventElapsedTime(&ms, a, b); best = ms < best ? ms : best;
    }
    return best;
}

static int ways(int op, int stride) {        // max distinct addresses per bank in a 32-lane half
    const int nb = BYTES[op] >= 4 ? BYTES[op] / 4 : 1;
    int worst = 1;
    for (int h = 0; h < 2; ++h) {
        int cnt[32] = {0};
        unsigned seen[32][64]; int ns[32] = {0};
        for (int l = 32 * h; l < 32 * h + 32; ++l) {
            const unsigned a = (unsigned)((l * stride) & (16383 & ~15));
            for (int k = 0; k < nb; ++k) {
                const unsigned word = a / 4 + k, b = word % 32;
                bool dup = false;
                for (int j = 0; j < ns[b]; ++j) dup |= seen[b][j] == word;
                if (!dup) { seen[b][ns[b]++] = word; cnt[b]++; }
            }
        }
        for (int b = 0; b < 32; ++b) worst = cnt[b] > worst ? cnt[b] : worst;
    }
    return worst;
}

template <int OP>
static void op_sweep(unsigned* out) {
    const int strides[] = {4, 8, 16, 32, 64, 128};
    for (int st : strides) {
        if (st < BYTES[OP] && OP != BPERM && OP != RB16) continue;
        const int iters = 2000, grid = 80 * 4;
        const float ms = best_ms([&] { hipLaunchKernelGGL(k_op<OP>, dim3(grid), dim3(256), 0, 0, iters, st, out); });
        const double insts = (double)grid * 4 * iters * 8;
        const double cyc = ms * 1e-3 * clk * 80 / insts;
        printf("LDSOP op=%s bytes=%d stride=%d ways=%d cyc_per_inst_cu=%.2f bytes_per_cyc_cu=%.1f\n", NAME[OP], BYTES[OP],
               st, OP == BPERM ? 1 : ways(OP, st), cyc, 64.0 * BYTES[OP] / cyc);
        fflush(stdout);
    }
}

template <int W>
static void xchg_sweep(unsigned* out) {
    for (int cp = 1; cp <= 3; ++cp) {
        for (int xb : {1024, 2048, 4096, 8192, 16384}) {
            const int lds = W * xb;
            if (lds * cp > 65536) continue;
            const int iters = 4000;
            const float ms = best_ms([&] { hipLaunchKernelGGL(k_xchg<W>, dim3(80 * cp), dim3(64 * W), lds, 0, iters, xb, out); });
            printf("XCHG W=%d ctas_per_cu=%d bytes_per_wave=%d lds=%d us_per_step=%.4f cyc_per_step=%.0f\n", W, cp, xb, lds,
                   ms * 1e3 / iters, ms * 1e-3 * clk / iters);
            fflush(stdout);
        }
    }
}

template <int W>
static void bar_sweep(unsigned* out) {
    for (int sk : {0, 16, 64, 256}) {
        const int iters = 20000;
        const float ms = best_ms([&] { hipLaunchKernelGGL(k_bar<W>, dim3(80), dim3(64 * W), 0, 0, iters, sk, out); });
        printf("BAR W=%d skew_valu=%d cyc_per_barrier=%.1f\n", W, sk, ms * 1e-3 * clk / iters);
        fflush(stdout);
    }
}

int main(int argc, char** argv) {
    if (argc > 1) clk = atof(argv[1]) * 1e9;
    unsigned* out; hipMalloc(&out, 80 * 4 * 1024 * 4);
    printf("# clk=%.3f GHz; LDSOP: 16 waves/CU, 8 ops/iter; ways = distinct addresses per bank per 32-lane half\n", clk / 1e9);
    op_sweep<RB32>(out); op_sweep<RB64>(out); op_sweep<RB128>(out); op_sweep<R2B32>(out); op_sweep<RB16>(out);
    op_sweep<WB32>(out); op_sweep<WB64>(out); op_sweep<WB128>(out); op_sweep<BPERM>(out);
    {
        const int iters = 20000;
        const float ms = best_ms([&] { hipLaunchKernelGGL(k_lat, dim3(80), dim3(64), 0, 0, iters, out); });
        printf("LDSLAT ds_read_b32 dependent chain: %.1f cycles\n", ms * 1e-3 * clk / iters);
    }
    xchg_sweep<4>(out); xchg_sweep<8>(out);
    bar_sweep<4>(out); bar_sweep<8>(out); bar_sweep<16>(out);
    return 0;
}
