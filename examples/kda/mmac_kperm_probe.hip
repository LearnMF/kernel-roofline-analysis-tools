// Is v_mmac_f32_16x16x16_bf16 invariant to a consistent permutation of the K index?
// Standard fragments: A lane (m, q) holds A[m][4q+e], B lane (n, q) holds B[4q+e][n].  Permuted: both
// operands take k = pi(4q+e) for the same permutation pi.  The exact sum is identical; if the
// hardware rounds partial sums in a fixed positional tree, results differ in the last bits.
// Bit-identical outputs over many random trials => K-permuted LDS layouts keep results bit-exact.
//   hipcc -O3 --offload-arch=gfx936 mmac_kperm_probe.hip -o kperm && ./kperm
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

typedef short v4bh __attribute__((ext_vector_type(4)));
typedef float f4 __attribute__((ext_vector_type(4)));

__global__ void k_perm(const short* A, const short* B, const float* C, const int* perm, float* D1, float* D2,
                       int ntile) {
    const int l = threadIdx.x & 63, w = (blockIdx.x * blockDim.x + threadIdx.x) >> 6;
    if (w >= ntile) return;
    const short* a = A + (size_t)w * 256;   // [16 m][16 k]
    const short* b = B + (size_t)w * 256;   // [16 k][16 n]
    const int* pi = perm + (size_t)w * 16;
    const int m = l & 15, q = l >> 4;
    v4bh af, bf, ap, bp;
    for (int e = 0; e < 4; ++e) {
        const int k = 4 * q + e, kp = pi[4 * q + e];
        af[e] = a[m * 16 + k]; bf[e] = b[k * 16 + m];
        ap[e] = a[m * 16 + kp]; bp[e] = b[kp * 16 + m];
    }
    f4 c;
    for (int e = 0; e < 4; ++e) c[e] = C[(size_t)w * 256 + l * 4 + e];
    const f4 d1 = __builtin_amdgcn_mmac_f32_16x16x16bf16(af, bf, c);
    const f4 d2 = __builtin_amdgcn_mmac_f32_16x16x16bf16(ap, bp, c);
    for (int e = 0; e < 4; ++e) {
        D1[(size_t)w * 256 + l * 4 + e] = d1[e];
        D2[(size_t)w * 256 + l * 4 + e] = d2[e];
    }
}

static short f2bf(float x) { uint32_t u; memcpy(&u, &x, 4); u += 0x7fff + ((u >> 16) & 1); return (short)(u >> 16); }

int main() {
    const int ntile = 1 << 15;   // 32768 tiles x 256 outputs = 8.4M dot products of 16 terms
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> ex(-12, 12);
    std::vector<short> A(ntile * 256), B(ntile * 256);
    std::vector<float> C(ntile * 256);
    std::vector<int> P(ntile * 16);
    for (auto& v : A) v = f2bf(nd(rng) * std::ldexp(1.f, ex(rng)));
    for (auto& v : B) v = f2bf(nd(rng) * std::ldexp(1.f, ex(rng)));
    for (int t = 0; t < ntile; ++t)
        for (int i = 0; i < 256; ++i) C[t * 256 + i] = (t % 3 == 0) ? 0.f : nd(rng) * std::ldexp(1.f, ex(rng));
    for (int t = 0; t < ntile; ++t) {
        int* pi = &P[t * 16];
        for (int i = 0; i < 16; ++i) pi[i] = i;
        if (t % 4 == 0) {           // the layout of interest: k = q + 4e  <->  4q + e
            for (int q = 0; q < 4; ++q) for (int e = 0; e < 4; ++e) pi[4 * q + e] = q + 4 * e;
        } else {
            std::shuffle(pi, pi + 16, rng);
        }
    }
    short *dA, *dB; float *dC, *dD1, *dD2; int* dP;
    hipMalloc(&dA, A.size() * 2); hipMalloc(&dB, B.size() * 2); hipMalloc(&dC, C.size() * 4);
    hipMalloc(&dP, P.size() * 4); hipMalloc(&dD1, C.size() * 4); hipMalloc(&dD2, C.size() * 4);
    hipMemcpy(dA, A.data(), A.size() * 2, hipMemcpyHostToDevice);
    hipMemcpy(dB, B.data(), B.size() * 2, hipMemcpyHostToDevice);
    hipMemcpy(dC, C.data(), C.size() * 4, hipMemcpyHostToDevice);
    hipMemcpy(dP, P.data(), P.size() * 4, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(k_perm, dim3(ntile / 4), dim3(256), 0, 0, dA, dB, dC, dP, dD1, dD2, ntile);
    hipDeviceSynchronize();
    std::vector<float> D1(C.size()), D2(C.size());
    hipMemcpy(D1.data(), dD1, D1.size() * 4, hipMemcpyDeviceToHost);
    hipMemcpy(D2.data(), dD2, D2.size() * 4, hipMemcpyDeviceToHost);
    long diff = 0, diff_t = 0, n_t = 0;
    for (int t = 0; t < ntile; ++t) {
        const bool tr = (t % 4 == 0);
        for (int i = 0; i < 256; ++i) {
            const bool d = memcmp(&D1[t * 256 + i], &D2[t * 256 + i], 4) != 0;
            diff += d;
            if (tr) { diff_t += d; ++n_t; }
        }
    }
    printf("outputs %zu: bitwise differences %ld (%.4f%%); transpose-permutation subset %ld of %ld\n",
           D1.size(), diff, 100.0 * diff / D1.size(), diff_t, n_t);
    return 0;
}
