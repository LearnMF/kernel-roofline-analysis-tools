// gfx938 MMAC lit / lts modifiers: semantics (vs lit = lts = 0 and vs a host 16x16x16 product under a few operand
// layout hypotheses) and cost (dependent-chain latency, independent-chain throughput).
//    hipcc -O3 --offload-arch=gfx938 mmac_lit.hip -o mmac_lit && ./mmac_lit
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

typedef short v4bh __attribute__((ext_vector_type(4)));
typedef float f4 __attribute__((ext_vector_type(4)));
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("%s\n", hipGetErrorString(e_)); exit(1); } } while (0)

template <bool LIT, bool LTS>
__device__ __forceinline__ f4 mm(v4bh a, v4bh b, f4 c) { return __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(a, b, c, LIT, LTS); }

// one wave: lane l holds A fragment a[l] (4 bf16) and B fragment b[l]; C starts at 0
template <bool LIT, bool LTS>
__global__ void k_sem(const v4bh* A, const v4bh* B, f4* C) {
    const int l = threadIdx.x;
    C[l] = mm<LIT, LTS>(A[l], B[l], f4{0.f, 0.f, 0.f, 0.f});
}

template <bool LIT, bool LTS, int NCH>
__global__ void __launch_bounds__(256) k_cost(int iters, float* sink) {
    v4bh a = v4bh{(short)(threadIdx.x | 0x3f00), 0x3f80, 0x3f80, 0x3f80}, b = a;
    f4 c[NCH];
    #pragma unroll
    for (int k = 0; k < NCH; ++k) c[k] = f4{0.f, 0.f, 0.f, (float)k};
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int k = 0; k < NCH; ++k) c[k] = mm<LIT, LTS>(a, b, c[k]);
    }
    float s = 0.f;
    #pragma unroll
    for (int k = 0; k < NCH; ++k) s += c[k][0] + c[k][3];
    if (s == 12345.f) *sink = s;
}

static float bf2f(short s) { unsigned u = (unsigned)(unsigned short)s << 16; float f; memcpy(&f, &u, 4); return f; }
static short f2bf(float f) { unsigned u; memcpy(&u, &f, 4); return (short)(u >> 16); }

template <bool LIT, bool LTS>
static void sem(const char* name, const std::vector<v4bh>& hA, const std::vector<v4bh>& hB, std::vector<f4>& out) {
    v4bh *A, *B; f4* C;
    CK(hipMalloc(&A, 64 * 8)); CK(hipMalloc(&B, 64 * 8)); CK(hipMalloc(&C, 64 * 16));
    CK(hipMemcpy(A, hA.data(), 64 * 8, hipMemcpyHostToDevice)); CK(hipMemcpy(B, hB.data(), 64 * 8, hipMemcpyHostToDevice));
    hipLaunchKernelGGL((k_sem<LIT, LTS>), dim3(1), dim3(64), 0, 0, A, B, C);
    out.resize(64); CK(hipMemcpy(out.data(), C, 64 * 16, hipMemcpyDeviceToHost));
    CK(hipFree(A)); CK(hipFree(B)); CK(hipFree(C));
    (void)name;
}

template <typename K>
static float run_k(K kern, int ctas, int iters, float* s) {
    hipLaunchKernelGGL(kern, dim3(ctas), dim3(256), 0, 0, iters, s); CK(hipDeviceSynchronize());
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    float best = 1e30f;
    for (int r = 0; r < 5; ++r) {
        CK(hipEventRecord(a)); hipLaunchKernelGGL(kern, dim3(ctas), dim3(256), 0, 0, iters, s);
        CK(hipEventRecord(b)); CK(hipEventSynchronize(b)); float ms; CK(hipEventElapsedTime(&ms, a, b)); if (ms < best) best = ms;
    }
    return best;
}
template <bool LIT, bool LTS>
static void cost(const char* name) {
    float* s; CK(hipMalloc(&s, 4));
    const int iters = 4096;
    // latency: 1 CTA (4 waves on 4 SIMDs), 1 chain -> cycles per dependent MMAC
    const float ml = run_k(k_cost<LIT, LTS, 1>, 1, iters, s);
    // throughput: 72 x 8 CTAs, 8 independent chains per wave
    const float mt = run_k(k_cost<LIT, LTS, 8>, 72 * 8, iters, s);
    const double flop = 2.0 * 16 * 16 * 16 * (double)iters * 8 * 72 * 8 * 4;
    printf("  %-14s dependent chain %6.1f cycles / MMAC   throughput %6.1f TFLOPS\n", name, ml * 1e-3 * 1.3e9 / iters, flop / (mt * 1e-3) / 1e12);
    CK(hipFree(s));
}

int main() {
    // operands: A[i][k], B[k][j] small integers (exact in bf16 and fp32); fragments per the gfx936 convention:
    // A frag lane l = (m = l & 15, q = l >> 4): a[e] = A[m][4q + e];  B frag lane l: b[e] = B[4q + e][m]  (n = m)
    float Am[16][16], Bm[16][16];
    srand(7);
    for (int i = 0; i < 16; ++i) for (int k = 0; k < 16; ++k) { Am[i][k] = (float)(rand() % 7 - 3); Bm[i][k] = (float)(rand() % 5 - 2); }
    std::vector<v4bh> hA(64), hB(64);
    for (int l = 0; l < 64; ++l) for (int e = 0; e < 4; ++e) {
        hA[l][e] = f2bf(Am[l & 15][4 * (l >> 4) + e]);
        hB[l][e] = f2bf(Bm[4 * (l >> 4) + e][l & 15]);
    }
    std::vector<f4> r00, r10, r01, r11;
    sem<false, false>("00", hA, hB, r00); sem<true, false>("10", hA, hB, r10);
    sem<false, true>("01", hA, hB, r01); sem<true, true>("11", hA, hB, r11);
    // reference products under layout hypotheses for the C fragment: lane l, element e -> (row, col)
    auto C = [&](int i, int j, bool tA, bool tB) {
        float s = 0.f;
        for (int k = 0; k < 16; ++k) s += (tA ? Am[k][i] : Am[i][k]) * (tB ? Bm[j][k] : Bm[k][j]);
        return s;
    };
    struct Hyp { const char* n; bool tA, tB; int map; };
    // map 0: (row 4q + e, col m)   map 1: (row m, col 4q + e)
    const Hyp hy[] = {{"A.B, C(4q+e, m)", false, false, 0}, {"A.B, C(m, 4q+e)", false, false, 1},
                      {"A^T.B, C(4q+e, m)", true, false, 0}, {"A.B^T, C(4q+e, m)", false, true, 0},
                      {"A^T.B^T, C(4q+e,m)", true, true, 0}, {"A^T.B, C(m,4q+e)", true, false, 1},
                      {"A.B^T, C(m, 4q+e)", false, true, 1}, {"A^T.B^T, C(m,4q+e)", true, true, 1}};
    const char* vn[] = {"lit0 lts0", "lit1 lts0", "lit0 lts1", "lit1 lts1"};
    std::vector<f4>* vr[] = {&r00, &r10, &r01, &r11};
    printf("semantics (16x16x16 bf16, integer operands; which reference each variant matches exactly):\n");
    for (int v = 0; v < 4; ++v) {
        printf("  %s:", vn[v]);
        int nm = 0;
        for (const Hyp& h : hy) {
            bool ok = true;
            for (int l = 0; l < 64 && ok; ++l) for (int e = 0; e < 4 && ok; ++e) {
                const int m = l & 15, q = l >> 4;
                const int i = h.map == 0 ? 4 * q + e : m, j = h.map == 0 ? m : 4 * q + e;
                if ((*vr[v])[l][e] != C(i, j, h.tA, h.tB)) ok = false;
            }
            if (ok) { printf(" [%s]", h.n); ++nm; }
        }
        int same = 0;
        for (int l = 0; l < 64; ++l) for (int e = 0; e < 4; ++e) same += (*vr[v])[l][e] == r00[l][e];
        printf("%s  (equal to lit0 lts0 at %d / 256 elements)\n", nm ? "" : " none", same);
    }
    printf("lane 0..3, element 0..3 of each variant:\n");
    for (int v = 0; v < 4; ++v) {
        printf("  %s:", vn[v]);
        for (int l = 0; l < 4; ++l) printf("  (%g %g %g %g)", (*vr[v])[l][0], (*vr[v])[l][1], (*vr[v])[l][2], (*vr[v])[l][3]);
        printf("\n");
    }
    printf("cost:\n");
    cost<false, false>("lit0 lts0"); cost<true, false>("lit1 lts0"); cost<false, true>("lit0 lts1"); cost<true, true>("lit1 lts1");
    return 0;
}
