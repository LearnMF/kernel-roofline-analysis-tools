// Is MMAC lit1 lts1 bit-identical to lit0 lts0 up to an output permutation, including fp32 accumulation?
// Step 1 maps every output position to the set of (A element, B element) products it sums, per mode (one-hot A,
// B[y] = y + 1, exact in bf16).  Step 2 matches the sets across modes.  Step 3 compares random runs bit for bit.
//    hipcc -O3 --offload-arch=gfx938 mmac_lit_exact.hip -o mmac_lit_exact && ./mmac_lit_exact
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>
typedef short v4bh __attribute__((ext_vector_type(4)));
typedef float f4 __attribute__((ext_vector_type(4)));
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("%s\n", hipGetErrorString(e_)); exit(1); } } while (0)

template <bool LIT, bool LTS>
__global__ void k(const v4bh* A, const v4bh* B, const f4* C, f4* D, int reps) {
    const int l = threadIdx.x;
    f4 c = C[l];
    for (int r = 0; r < reps; ++r) c = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(A[l], B[l], c, LIT, LTS);
    D[l] = c;
}
static short f2bf(float f) { unsigned u; memcpy(&u, &f, 4); return (short)((u + 0x7fff + ((u >> 16) & 1)) >> 16); }
struct Dev { v4bh *A, *B; f4 *C, *D; };
template <bool LIT, bool LTS>
static void run(Dev& d, const v4bh* a, const v4bh* b, const f4* c, f4* out, int reps) {
    CK(hipMemcpy(d.A, a, 512, hipMemcpyHostToDevice)); CK(hipMemcpy(d.B, b, 512, hipMemcpyHostToDevice));
    CK(hipMemcpy(d.C, c, 1024, hipMemcpyHostToDevice));
    hipLaunchKernelGGL((k<LIT, LTS>), dim3(1), dim3(64), 0, 0, d.A, d.B, d.C, d.D, reps);
    CK(hipMemcpy(out, d.D, 1024, hipMemcpyDeviceToHost));
}
template <bool LIT, bool LTS>
static std::vector<std::vector<std::pair<int, int>>> signature(Dev& d) {
    std::vector<std::vector<std::pair<int, int>>> sig(256);
    v4bh a[64], b[64]; f4 z[64], o[64];
    for (int l = 0; l < 64; ++l) for (int e = 0; e < 4; ++e) { b[l][e] = f2bf((float)(4 * l + e + 1)); z[l][e] = 0.f; }
    for (int x = 0; x < 256; ++x) {
        for (int l = 0; l < 64; ++l) for (int e = 0; e < 4; ++e) a[l][e] = (4 * l + e == x) ? f2bf(1.f) : (short)0;
        run<LIT, LTS>(d, a, b, z, o, 1);
        for (int p = 0; p < 256; ++p) {
            const float v = o[p / 4][p % 4];
            if (v != 0.f) sig[p].push_back({x, (int)v - 1});
        }
    }
    for (auto& s : sig) std::sort(s.begin(), s.end());
    return sig;
}
int main() {
    Dev d; CK(hipMalloc(&d.A, 512)); CK(hipMalloc(&d.B, 512)); CK(hipMalloc(&d.C, 1024)); CK(hipMalloc(&d.D, 1024));
    auto s00 = signature<false, false>(d), s11 = signature<true, true>(d);
    std::map<std::vector<std::pair<int, int>>, int> idx;
    for (int p = 0; p < 256; ++p) idx[s00[p]] = p;
    std::vector<int> perm(256, -1); int sz_ok = 0, found = 0;
    for (int p = 0; p < 256; ++p) {
        sz_ok += s00[p].size() == 16 && s11[p].size() == 16;
        auto it = idx.find(s11[p]); if (it != idx.end()) { perm[p] = it->second; ++found; }
    }
    printf("every output sums 16 products in both modes: %d / 256;  lit1lts1 positions matched to a lit0lts0 position: %d / 256\n", sz_ok, found);
    printf("perm lit1lts1 (lane, e) -> lit0lts0 (lane, e), lanes 0..2: ");
    for (int p = 0; p < 12; ++p) printf("(%d,%d)->(%d,%d) ", p / 4, p % 4, perm[p] / 4, perm[p] % 4);
    printf("\n");
    srand(3); auto rnd = [] { return (float)rand() / RAND_MAX * 2.f - 1.f; };
    v4bh a[64], b[64]; f4 c0[64], c1[64], r00[64], r11[64];
    for (int t = 0; t < 3; ++t) {
        for (int l = 0; l < 64; ++l) for (int e = 0; e < 4; ++e) { a[l][e] = f2bf(rnd()); b[l][e] = f2bf(rnd()); c0[l][e] = rnd() * 4.f; }
        for (int p = 0; p < 256; ++p) c1[p / 4][p % 4] = perm[p] >= 0 ? c0[perm[p] / 4][perm[p] % 4] : 0.f;
        for (int reps : {1, 8, 64}) {
            run<false, false>(d, a, b, c0, r00, reps); run<true, true>(d, a, b, c1, r11, reps);
            int eq = 0;
            for (int p = 0; p < 256; ++p) if (perm[p] >= 0) {
                float fx = r11[p / 4][p % 4], fy = r00[perm[p] / 4][perm[p] % 4]; unsigned x, y; memcpy(&x, &fx, 4); memcpy(&y, &fy, 4); eq += x == y;
            }
            printf("trial %d, %2d chained MMACs (random bf16 A / B, random fp32 C): bit-identical %d / 256\n", t, reps, eq);
        }
    }
    return 0;
}
