// LDS bank structure on gfx936 / gfx938: cycles per ds_read / ds_write instruction (one wave per CU, 64 lanes) for
// lane address = lane * stride (bytes), widths b32 / b64 / b128, plus the row-pattern layouts used by KDA kernels.
// Conflict degree = cycles / cycles at the conflict-free stride.
//    hipcc -O3 --offload-arch=gfx938 lds_bank.hip -o lds_bank && ./lds_bank
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("%s\n", hipGetErrorString(e_)); exit(1); } } while (0)
typedef int i2v __attribute__((ext_vector_type(2)));
typedef int i4v __attribute__((ext_vector_type(4)));
constexpr int ITERS = 256, UNR = 8, LDS_BYTES = 65536;

template <int W, bool WR>
__global__ void __launch_bounds__(64) k_lds(const int* __restrict__ off, long long* out, int* sink) {
    extern __shared__ __align__(16) char lds[];
    const int l = threadIdx.x;
    for (int i = l; i < LDS_BYTES / 4; i += 64) reinterpret_cast<int*>(lds)[i] = i;
    __syncthreads();
    const int o = off[l];
    int acc = 0;
    long long t0 = 0, t1 = 0;
    for (int rep = 0; rep < 2; ++rep) {                      // rep 0 warms the instruction cache
        __builtin_amdgcn_s_waitcnt(0);
        t0 = __builtin_amdgcn_s_memtime();
        for (int it = 0; it < ITERS; ++it) {
            #pragma unroll
            for (int u = 0; u < UNR; ++u) {
                char* p = lds + ((o + u * 0) & (LDS_BYTES - 1));
                if constexpr (WR) {
                    if constexpr (W == 4) *reinterpret_cast<volatile int*>(p) = acc + u;
                    else if constexpr (W == 8) *reinterpret_cast<volatile i2v*>(p) = i2v{acc, u};
                    else *reinterpret_cast<volatile i4v*>(p) = i4v{acc, u, it, l};
                } else {
                    if constexpr (W == 4) acc += *reinterpret_cast<volatile int*>(p);
                    else if constexpr (W == 8) { i2v v = *reinterpret_cast<volatile i2v*>(p); acc += v.x ^ v.y; }
                    else { i4v v = *reinterpret_cast<volatile i4v*>(p); acc += v.x ^ v.y ^ v.z ^ v.w; }
                }
            }
        }
        __builtin_amdgcn_s_waitcnt(0);
        t1 = __builtin_amdgcn_s_memtime();
    }
    if (l == 0) *out = t1 - t0;
    if (acc == 0x7fffffff) *sink = acc;
}

template <int W, bool WR>
static double run(const int* hoff, int* doff, long long* dout, int* dsink) {
    CK(hipMemcpy(doff, hoff, 64 * 4, hipMemcpyHostToDevice));
    double best = 1e30;
    for (int r = 0; r < 5; ++r) {
        hipLaunchKernelGGL((k_lds<W, WR>), dim3(1), dim3(64), LDS_BYTES, 0, doff, dout, dsink);
        long long c; CK(hipMemcpy(&c, dout, 8, hipMemcpyDeviceToHost));
        const double per = (double)c / (ITERS * UNR);
        if (per < best) best = per;
    }
    return best;   // s_memtime ticks per instruction (constant-rate counter; compare ratios)
}

int main() {
    int *doff, *dsink; long long* dout;
    CK(hipMalloc(&doff, 64 * 4)); CK(hipMalloc(&dout, 8)); CK(hipMalloc(&dsink, 4));
    int hoff[64];
    const int strides[] = {4, 8, 12, 16, 20, 24, 32, 36, 48, 64, 68, 72, 80, 96, 128, 132, 136, 144, 192, 256, 264, 272, 512, 1024};
    printf("lane address = lane * stride; ticks per instruction (read / write) for b32, b64, b128\n");
    printf("%7s | %8s %8s | %8s %8s | %8s %8s\n", "stride", "r32", "w32", "r64", "w64", "r128", "w128");
    for (int s : strides) {
        for (int l = 0; l < 64; ++l) hoff[l] = l * s;
        double r4 = run<4, false>(hoff, doff, dout, dsink), w4 = run<4, true>(hoff, doff, dout, dsink);
        double r8 = (s % 8) ? -1 : run<8, false>(hoff, doff, dout, dsink), w8 = (s % 8) ? -1 : run<8, true>(hoff, doff, dout, dsink);
        double r16 = (s % 16) ? -1 : run<16, false>(hoff, doff, dout, dsink), w16 = (s % 16) ? -1 : run<16, true>(hoff, doff, dout, dsink);
        printf("%7d | %8.2f %8.2f | %8.2f %8.2f | %8.2f %8.2f\n", s, r4, w4, r8, w8, r16, w16);
    }
    // KDA-style 2D patterns: lane (m = l & 15, q = l >> 4) reads row m (pitch P bytes), column group q (G bytes)
    printf("\n2D pattern: lane (m = l & 15, q = l >> 4) -> m * pitch + q * G; ticks per instruction (read b64 / read b128)\n");
    const int pitches[] = {128, 136, 144, 256, 264, 272, 512, 528, 544};
    for (int P : pitches) {
        for (int l = 0; l < 64; ++l) hoff[l] = (l & 15) * P + (l >> 4) * 8;
        double a = run<8, false>(hoff, doff, dout, dsink);
        for (int l = 0; l < 64; ++l) hoff[l] = (l & 15) * P + (l >> 4) * 16;
        double b = (P % 16) ? -1 : run<16, false>(hoff, doff, dout, dsink);
        printf("  pitch %4d B: b64 (G = 8) %6.2f   b128 (G = 16) %6.2f\n", P, a, b);
    }
    return 0;
}
