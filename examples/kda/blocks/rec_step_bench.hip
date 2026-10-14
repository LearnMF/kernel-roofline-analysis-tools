// Building-block library P0-1: parametric recurrence step on gfx936 (composition of MMAC / VALU / LDS / barrier /
// VMEM inside one serial step).  One loop iteration = one step with the production dependency structure:
//   [VMEM loads for the next step (prefetch) | or this step]
//   GEMM1: n1 MMAC in i1 (2 or 4, compile-time) independent accumulator chains
//   VALU block 1 (nv1 v_fma, depends on GEMM1)  -> nw1 ds_write_b64 -> s_barrier -> nr ds_read_b64
//   GEMM2: n2 MMAC in the same number of chains (i2 is ignored; B operand = the LDS values)
//   VALU block 2 (nv2, depends on GEMM2) -> nw2 ds_write_b64 -> s_barrier
//   nst VMEM stores (state emission)
// Counts are per wave per step; every block is unrolled by 4 around a runtime count.  W waves per CTA, CTAs/CU.
//   hipcc -O3 --offload-arch=gfx936 rec_step_bench.hip -o recb && ./recb [config file] > rec_step_bench.txt
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

typedef short v4bh __attribute__((ext_vector_type(4)));
typedef float f4 __attribute__((ext_vector_type(4)));
typedef unsigned u2 __attribute__((ext_vector_type(2)));

struct Cfg {
    int n1, i1, n2, i2, nv1, nv2, nw1, nw2, nr, nld, nst, bar, pf, hbm;
};

__device__ __forceinline__ f4 mm(v4bh a, v4bh b, f4 c) { return __builtin_amdgcn_mmac_f32_16x16x16bf16(a, b, c); }

template <int W, int G>
__global__ void __launch_bounds__(64 * W) k_step(Cfg c, int NT, const char* __restrict__ g, char* __restrict__ gs,
                                                 size_t gmask, float* out) {
    __shared__ __align__(16) unsigned char lds[16384];
    const int tid = threadIdx.x, l = tid & 63, w = tid >> 6;
    const unsigned lbase = (unsigned)(uintptr_t)lds + (unsigned)(w * 2048 + l * 8);
    const unsigned rbase = (unsigned)(uintptr_t)lds + (unsigned)(((w + 1) % W) * 2048 + l * 8);
    v4bh a = v4bh{(short)(0x3c00 + l), 0x3c00, 0x3c01, 0x3c02};
    f4 acc[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = f4{0.f, 0.f, 0.f, (float)i};
    float x[4] = {1.f, 2.f, 3.f, 4.f};
    u2 ldv[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) ldv[i] = u2{0u, 0u};
    // each CTA walks its own region; one step's loads = nld x 512 B per wave (native 8 B / lane fragments)
    const size_t cta_off = ((size_t)blockIdx.x * 8191u * 65536u) & gmask;
    for (int s = 0; s < NT; ++s) {
        const size_t soff = (cta_off + (size_t)s * (size_t)(c.nld * 512 * W + 4096)) & gmask;
        // ---- VMEM loads (prefetch: consumed next step; pf = 0: waited for right away)
        if (c.pf) asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        unsigned use = 0;
        #pragma unroll
        for (int i = 0; i < 16; ++i) if (i < c.nld) use ^= ldv[i].x ^ ldv[i].y;
        #pragma unroll
        for (int i = 0; i < 16; ++i)
            if (i < c.nld) ldv[i] = *reinterpret_cast<const u2*>(g + soff + (size_t)(w * c.nld + i) * 512 + l * 8);
        if (!c.pf) asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        a[0] = (short)(a[0] ^ (short)(use & 1));
        // ---- GEMM1
        for (int i = 0; i < c.n1; i += 8) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) if (i + j < c.n1) acc[j % G] = mm(a, a, acc[j % G]);
        }
        // ---- VALU block 1 (depends on GEMM1)
        float y = acc[0][0] + acc[G - 1][1];
        for (int i = 0; i < c.nv1; i += 4) {
            asm volatile("v_fma_f32 %0, %0, %1, 1.0\n v_fma_f32 %2, %2, %1, 1.0\n v_fma_f32 %3, %3, %1, 1.0\n v_fma_f32 %4, %4, %1, 1.0"
                         : "+v"(x[0]), "+v"(y), "+v"(x[1]), "+v"(x[2]));
            asm volatile("" : "+v"(x[3]));
        }
        // ---- LDS write -> barrier -> LDS read
        const unsigned wv = __float_as_uint(y + x[0]);
        for (int i = 0; i < c.nw1; i += 4) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) if (i + j < c.nw1) { u2 v = {wv, (unsigned)j}; asm volatile("ds_write_b64 %0, %1" :: "v"(lbase + (unsigned)j * 512u), "v"(v)); }
        }
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        if (c.bar >= 1) asm volatile("s_barrier" ::: "memory");
        unsigned rd = 0;
        for (int i = 0; i < c.nr; i += 4) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) if (i + j < c.nr) { u2 v; asm volatile("ds_read_b64 %0, %1" : "=v"(v) : "v"(rbase + (unsigned)j * 512u)); rd ^= v.x; }
        }
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        v4bh b = a;
        b[1] = (short)(b[1] ^ (short)(rd & 1));
        // ---- GEMM2 (depends on the LDS values)
        for (int i = 0; i < c.n2; i += 8) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) if (i + j < c.n2) acc[j % G] = mm(b, a, acc[j % G]);
        }
        // ---- VALU block 2 -> LDS write -> barrier
        float z = acc[0][2];
        for (int i = 0; i < c.nv2; i += 4) {
            asm volatile("v_fma_f32 %0, %0, %1, 1.0\n v_fma_f32 %2, %2, %1, 1.0\n v_fma_f32 %3, %3, %1, 1.0\n v_fma_f32 %4, %4, %1, 1.0"
                         : "+v"(x[3]), "+v"(z), "+v"(x[1]), "+v"(x[2]));
        }
        const unsigned zv = __float_as_uint(z);
        for (int i = 0; i < c.nw2; i += 4) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) if (i + j < c.nw2) { u2 v = {zv, (unsigned)j}; asm volatile("ds_write_b64 %0, %1" :: "v"(lbase + 8192u + (unsigned)j * 512u), "v"(v)); }
        }
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        if (c.bar >= 2) asm volatile("s_barrier" ::: "memory");
        // ---- state stores
        #pragma unroll
        for (int i = 0; i < 16; ++i)
            if (i < c.nst) *reinterpret_cast<u2*>(gs + soff + (size_t)(w * c.nst + i) * 512 + l * 8) = u2{zv, (unsigned)i};
        a[2] = (short)(a[2] ^ (short)(zv & 1));
    }
    float s = x[0] + x[1] + x[2] + x[3];
    #pragma unroll
    for (int i = 0; i < 8; ++i) s += acc[i][0];
    if (s == 1.2345f) out[blockIdx.x * 64 * W + tid] = s;
}

static double clk = 1.5e9;

template <int W, int G>
static float run(const Cfg& c, int cp, const char* g, char* gs, size_t gmask, float* out, int NT) {
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    hipLaunchKernelGGL((k_step<W, G>), dim3(80 * cp), dim3(64 * W), 0, 0, c, 8, g, gs, gmask, out);
    hipDeviceSynchronize();
    float best = 1e30f;
    for (int r = 0; r < 3; ++r) {
        hipEventRecord(e0);
        hipLaunchKernelGGL((k_step<W, G>), dim3(80 * cp), dim3(64 * W), 0, 0, c, NT, g, gs, gmask, out);
        hipEventRecord(e1); hipEventSynchronize(e1);
        float ms; hipEventElapsedTime(&ms, e0, e1); best = ms < best ? ms : best;
    }
    return best * 1e3f / NT;    // us per step (every CTA runs its own chain concurrently)
}

int main(int argc, char** argv) {
    const size_t big = (size_t)1 << 30;
    char *g, *gs; float* out;
    hipMalloc(&g, big + (1 << 22)); hipMalloc(&gs, big + (1 << 22)); hipMalloc(&out, 80 * 3 * 512 * 4);
    hipMemset(g, 0, big + (1 << 22));
    FILE* f = argc > 1 ? fopen(argv[1], "r") : stdin;
    char line[512];
    printf("# clk %.2f GHz; us per step; counts per wave per step\n", clk / 1e9);
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || strlen(line) < 5) continue;
        char tag[64]; int W, cp; Cfg c;
        if (sscanf(line, "%63s %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d", tag, &W, &cp, &c.n1, &c.i1, &c.n2, &c.i2,
                   &c.nv1, &c.nv2, &c.nw1, &c.nw2, &c.nr, &c.nld, &c.nst, &c.bar, &c.pf, &c.hbm) != 17) continue;
        const size_t gmask = (c.hbm ? big : ((size_t)4 << 20)) - 1;
        const int NT = 512;
        const float us = W == 8 ? (c.i1 >= 4 ? run<8, 4>(c, cp, g, gs, gmask, out, NT) : run<8, 2>(c, cp, g, gs, gmask, out, NT))
                                 : (c.i1 >= 4 ? run<4, 4>(c, cp, g, gs, gmask, out, NT) : run<4, 2>(c, cp, g, gs, gmask, out, NT));
        printf("STEP %s W=%d cp=%d n1=%d i1=%d n2=%d i2=%d nv1=%d nv2=%d nw1=%d nw2=%d nr=%d nld=%d nst=%d bar=%d pf=%d hbm=%d us=%.4f cyc=%.0f\n",
               tag, W, cp, c.n1, c.i1, c.n2, c.i2, c.nv1, c.nv2, c.nw1, c.nw2, c.nr, c.nld, c.nst, c.bar, c.pf, c.hbm, us, us * 1e-6 * clk);
        fflush(stdout);
    }
    return 0;
}
