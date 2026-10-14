// Summation semantics of the gfx936 MMAC accumulators, to decide whether a shfl reduction can be replaced by an
// MMAC "multiply by ones" reduction WITHOUT changing bits.
//   f32 K4  : C[m][n] = c + sum_{k<4}  A[m][k] B[n][k]    (A: lane (q,m) holds A[m][q], 1 fp32)
//   bf16 K16: C[m][n] = c + sum_{k<16} A[m][k] B[n][k]    (A: lane (q,m) holds A[m][4q..4q+3])
// With B = ones, row m of C is the MMAC's own sum of the four (sixteen) A values; compare it against
// sequential / pairwise-tree / exact (double, rounded once) sums of the same values.
//   hipcc -O3 --offload-arch=gfx936 mmac_sum_sem.hip -o mss && ./mss
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
typedef float f4 __attribute__((ext_vector_type(4)));
typedef short v4bh __attribute__((ext_vector_type(4)));

__global__ void k_f32(const float* x, const float* c0, float* out, int ntile) {
    const int l = threadIdx.x, t = blockIdx.x;
    if (t >= ntile) return;
    const float a = x[t * 64 + l];                     // A[m][q] with m = l & 15, q = l >> 4
    f4 c = {c0[t * 16 + (l & 15)], c0[t * 16 + (l & 15)], c0[t * 16 + (l & 15)], c0[t * 16 + (l & 15)]};
    f4 r = __builtin_amdgcn_mmac_f32_16x16x4f32(a, 1.0f, c);
    if (l < 16) out[t * 16 + l] = r[0];                // lane (0, m), element 0 = C[m][0]
}
__global__ void k_bf16(const short* x, const float* c0, float* out, int ntile) {
    const int l = threadIdx.x, t = blockIdx.x;
    if (t >= ntile) return;
    v4bh a; for (int e = 0; e < 4; ++e) a[e] = x[t * 256 + l * 4 + e];   // A[m][4q+e]
    v4bh one = {0x3f80, 0x3f80, 0x3f80, 0x3f80};
    const float cc = c0[t * 16 + (l & 15)];
    f4 c = {cc, cc, cc, cc};
    f4 r = __builtin_amdgcn_mmac_f32_16x16x16bf16(a, one, c);
    if (l < 16) out[t * 16 + l] = r[0];
}
static float bf2f(short s) { unsigned u = ((unsigned)(unsigned short)s) << 16; float f; memcpy(&f, &u, 4); return f; }
static float rnd(unsigned& s) { s = s * 1664525u + 1013904223u; return (s >> 8) * (1.0f / 16777216.0f); }

int main() {
    const int NT = 4096;
    std::vector<float> x(NT * 64), c0(NT * 16), o(NT * 16);
    std::vector<short> xb(NT * 256);
    unsigned s = 12345;
    for (auto& v : x) v = (rnd(s) - 0.5f) * std::ldexp(1.0f, (int)(rnd(s) * 40) - 20);
    for (auto& v : c0) v = (rnd(s) < 0.5f) ? 0.f : (rnd(s) - 0.5f) * std::ldexp(1.0f, (int)(rnd(s) * 40) - 20);
    for (auto& v : xb) { float f = (rnd(s) - 0.5f) * std::ldexp(1.0f, (int)(rnd(s) * 30) - 15); unsigned u; memcpy(&u, &f, 4); v = (short)(u >> 16); }
    float *dx, *dc, *dout; short* dxb;
    hipMalloc(&dx, x.size() * 4); hipMalloc(&dc, c0.size() * 4); hipMalloc(&dout, o.size() * 4); hipMalloc(&dxb, xb.size() * 2);
    hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice);
    hipMemcpy(dc, c0.data(), c0.size() * 4, hipMemcpyHostToDevice);
    hipMemcpy(dxb, xb.data(), xb.size() * 2, hipMemcpyHostToDevice);
    // ---- f32 K4
    hipLaunchKernelGGL(k_f32, dim3(NT), dim3(64), 0, 0, dx, dc, dout, NT);
    hipMemcpy(o.data(), dout, o.size() * 4, hipMemcpyDeviceToHost);
    long n = 0, seq = 0, seqc = 0, tree = 0, treec = 0, exact = 0;
    for (int t = 0; t < NT; ++t) for (int m = 0; m < 16; ++m) {
        float v[4]; for (int q = 0; q < 4; ++q) v[q] = x[t * 64 + 16 * q + m];
        const float c = c0[t * 16 + m], r = o[t * 16 + m];
        float a1 = c; for (int q = 0; q < 4; ++q) a1 += v[q];                 // c first, sequential
        float a2 = v[0]; for (int q = 1; q < 4; ++q) a2 += v[q]; a2 += c;     // sequential, c last
        float a3 = c + ((v[0] + v[1]) + (v[2] + v[3]));                       // xor-16/32 butterfly tree, c last
        float a4 = ((v[0] + v[1]) + (v[2] + v[3])) ; a4 = c + a4;
        double d = c; for (int q = 0; q < 4; ++q) d += v[q];
        ++n; seq += a1 == r; seqc += a2 == r; tree += a3 == r; treec += a4 == r; exact += (float)d == r;
    }
    printf("f32 K4   (%ld rows): c-first seq %ld | seq c-last %ld | tree c-last %ld | exact-once %ld\n", n, seq, seqc, tree, exact);
    // ---- bf16 K16
    hipLaunchKernelGGL(k_bf16, dim3(NT), dim3(64), 0, 0, dxb, dc, dout, NT);
    hipMemcpy(o.data(), dout, o.size() * 4, hipMemcpyDeviceToHost);
    n = seq = seqc = tree = exact = 0; long q4tree = 0;
    for (int t = 0; t < NT; ++t) for (int m = 0; m < 16; ++m) {
        float v[16]; for (int q = 0; q < 4; ++q) for (int e = 0; e < 4; ++e) v[4 * q + e] = bf2f(xb[t * 256 + (16 * q + m) * 4 + e]);
        const float c = c0[t * 16 + m], r = o[t * 16 + m];
        float a1 = c; for (int k = 0; k < 16; ++k) a1 += v[k];
        float a2 = v[0]; for (int k = 1; k < 16; ++k) a2 += v[k]; a2 += c;
        float w[16]; for (int k = 0; k < 16; ++k) w[k] = v[k];
        for (int st = 1; st < 16; st *= 2) for (int k = 0; k < 16; k += 2 * st) w[k] = w[k] + w[k + st];
        float a3 = c + w[0];
        float g[4]; for (int q = 0; q < 4; ++q) g[q] = ((v[4 * q] + v[4 * q + 1]) + (v[4 * q + 2] + v[4 * q + 3]));
        float a5 = c + ((g[0] + g[1]) + (g[2] + g[3]));
        double d = c; for (int k = 0; k < 16; ++k) d += v[k];
        ++n; seq += a1 == r; seqc += a2 == r; tree += a3 == r; q4tree += a5 == r; exact += (float)d == r;
    }
    printf("bf16 K16 (%ld rows): c-first seq %ld | seq c-last %ld | pairwise tree %ld | quad-tree %ld | exact-once %ld\n",
           n, seq, seqc, tree, q4tree, exact);
    return 0;
}
