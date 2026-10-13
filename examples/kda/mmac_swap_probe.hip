// F3a pre-check: is D = X . Y^T (A operand = X rows, B operand = Y rows) bit-identical to the transpose of
// E = Y . X^T (operands swapped), with the dav accumulation (8 K-steps of 16, from zero, then * scale)?
// The products are the same and the K order is the same; only the operand roles differ.
//   hipcc -O3 --offload-arch=gfx936 mmac_swap_probe.hip -o swp && ./swp
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

typedef short v4bh __attribute__((ext_vector_type(4)));
typedef float f4 __attribute__((ext_vector_type(4)));

// X, Y: [16 rows][128 k] bf16 per tile.  Lane (m, q): row m, k quad 16vs + 4q.
__global__ void k_swap(const short* X, const short* Y, float* D, float* E, int ntile, float scale) {
    const int l = threadIdx.x & 63, w = (blockIdx.x * blockDim.x + threadIdx.x) >> 6;
    if (w >= ntile) return;
    const short* x = X + (size_t)w * 2048;
    const short* y = Y + (size_t)w * 2048;
    const int m = l & 15, q = l >> 4;
    f4 d = {0.f, 0.f, 0.f, 0.f}, e = {0.f, 0.f, 0.f, 0.f};
    for (int vs = 0; vs < 8; ++vs) {
        v4bh xf, yf;
        for (int i = 0; i < 4; ++i) { xf[i] = x[m * 128 + 16 * vs + 4 * q + i]; yf[i] = y[m * 128 + 16 * vs + 4 * q + i]; }
        d = __builtin_amdgcn_mmac_f32_16x16x16bf16(xf, yf, d);   // D[m][q + 4e] = sum X[m][k] Y[q+4e][k]
        e = __builtin_amdgcn_mmac_f32_16x16x16bf16(yf, xf, e);   // E[m][q + 4e] = sum Y[m][k] X[q+4e][k]
    }
    for (int i = 0; i < 4; ++i) {
        D[(size_t)w * 256 + m * 16 + q + 4 * i] = d[i] * scale;   // D[row][col]
        E[(size_t)w * 256 + m * 16 + q + 4 * i] = e[i] * scale;
    }
}

static short f2bf(float x) { uint32_t u; memcpy(&u, &x, 4); u += 0x7fff + ((u >> 16) & 1); return (short)(u >> 16); }

int main() {
    const int ntile = 1 << 14;
    std::mt19937 rng(77);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> ex(-10, 10);
    std::vector<short> X(ntile * 2048), Y(ntile * 2048);
    for (auto& v : X) v = f2bf(nd(rng) * std::ldexp(1.f, ex(rng)));
    for (auto& v : Y) v = f2bf(nd(rng) * std::ldexp(1.f, ex(rng)));
    short *dX, *dY; float *dD, *dE;
    hipMalloc(&dX, X.size() * 2); hipMalloc(&dY, Y.size() * 2);
    hipMalloc(&dD, ntile * 256 * 4); hipMalloc(&dE, ntile * 256 * 4);
    hipMemcpy(dX, X.data(), X.size() * 2, hipMemcpyHostToDevice);
    hipMemcpy(dY, Y.data(), Y.size() * 2, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(k_swap, dim3(ntile / 4), dim3(256), 0, 0, dX, dY, dD, dE, ntile, 0.08838834764831845f);
    hipDeviceSynchronize();
    std::vector<float> D(ntile * 256), E(ntile * 256);
    hipMemcpy(D.data(), dD, D.size() * 4, hipMemcpyDeviceToHost);
    hipMemcpy(E.data(), dE, E.size() * 4, hipMemcpyDeviceToHost);
    long diff = 0;
    for (int t = 0; t < ntile; ++t)
        for (int r = 0; r < 16; ++r)
            for (int c = 0; c < 16; ++c)
                diff += memcmp(&D[t * 256 + r * 16 + c], &E[t * 256 + c * 16 + r], 4) != 0;
    printf("outputs %d: D[r][c] vs E[c][r] bitwise differences %ld\n", ntile * 256, diff);
    return 0;
}
