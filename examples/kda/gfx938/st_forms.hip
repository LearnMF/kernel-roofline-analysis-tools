// Store cost per access form on gfx938 (time + optional hipprof PMC TA_TA_BUSY_sum).
// Every form writes the same number of useful bytes (BYTES) with 256-thread CTAs, CTAS CTAs; the CTA count
// and a dynamic LDS pad set the residency like the production kernels (2-4 CTAs per CU).
//   0  16 B / lane, contiguous            (a wave instruction writes 1 KB)
//   1   8 B / lane, contiguous            (512 B)
//   2   4 B / lane, contiguous            (256 B)
//   3  16 B / lane, 256 B row segments    (4 rows per instruction, row stride RS: a head's [128] bf16 slice)
//   4   8 B / lane,  64 B row segments    (8 rows per instruction: a 32-column bf16 o slab)
//   5  16 B / lane, contiguous, nontemporal
//   6   2 B / lane, 4 scattered rows       (the pre-P-938l dhu aN emission)
//   7  16 B / lane, 64 B row segments, nontemporal
//   8  16 B / lane, 256 B row segments, nontemporal
//   9   8 B / lane, 32 B row segments, 16 rows per instruction; 4 consecutive instructions complete each 128 B row
//       (the prep_b Aqk / Akk row-major emit: [T, H, 64] bf16, row stride 8 KB at H = 64)
//  10  16 B / lane, 128 B row segments (8 lanes per row, 8 rows per instruction), same rows as 9
//    hipcc -O3 --offload-arch=gfx938 st_forms.hip -o st_forms && ./st_forms [form ...]
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

typedef unsigned int u4 __attribute__((ext_vector_type(4)));
typedef unsigned int u2 __attribute__((ext_vector_type(2)));
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("%s\n", hipGetErrorString(e_)); exit(1); } } while (0)

constexpr int CTAS = 8192;                  // the 8192/64 grid
constexpr size_t PER_CTA = 64 * 1024;       // 64 KB useful bytes per CTA -> 512 MB per launch
constexpr size_t RS = 16384;                // row stride of [T, 64, 128] bf16

template <int F>
__global__ void __launch_bounds__(256) k_st(char* __restrict__ dst, unsigned salt) {
    extern __shared__ char pad[];
    if (salt == 0xffffffffu) pad[threadIdx.x] = 0;                 // keeps the dynamic LDS (residency knob)
    const int l = threadIdx.x & 63, w = threadIdx.x >> 6;
    const size_t cta = blockIdx.x;
    const unsigned v = salt ^ (unsigned)(cta * 2654435761u) ^ threadIdx.x;
    if constexpr (F == 0 || F == 5) {                              // 16 B contiguous: 64 KB = 16 iterations x 4 waves x 1 KB
        char* b = dst + cta * PER_CTA;
        #pragma unroll 4
        for (int it = 0; it < 16; ++it) {
            u4* p = reinterpret_cast<u4*>(b + ((size_t)it * 4 + w) * 1024) + l;
            const u4 x = u4{v, v + it, v ^ 3u, v + 7u};
            if constexpr (F == 5) __builtin_nontemporal_store(x, p); else *p = x;
        }
    } else if constexpr (F == 1) {
        char* b = dst + cta * PER_CTA;
        #pragma unroll 4
        for (int it = 0; it < 32; ++it) *(reinterpret_cast<u2*>(b + ((size_t)it * 4 + w) * 512) + l) = u2{v, v + it};
    } else if constexpr (F == 2) {
        char* b = dst + cta * PER_CTA;
        #pragma unroll 4
        for (int it = 0; it < 64; ++it) *(reinterpret_cast<unsigned*>(b + ((size_t)it * 4 + w) * 256) + l) = v + it;
    } else if constexpr (F == 3 || F == 8) {                       // 256 B per row: lanes 16 per row, 4 rows per instruction
        // CTA owns 256 rows x 256 B (a 64-row chunk of 4 heads ... the slice pattern of a row-major [T, H, 128] tensor)
        char* b = dst + (cta / 64) * (256 * RS) + (cta % 64) * 256;
        #pragma unroll 4
        for (int it = 0; it < 16; ++it) {
            const int row = (it * 4 + w) * 4 + (l >> 4);
            u4* p = reinterpret_cast<u4*>(b + (size_t)row * RS) + (l & 15);
            const u4 x = u4{v, v + it, v ^ 3u, v + 7u};
            if constexpr (F == 8) __builtin_nontemporal_store(x, p); else *p = x;
        }
    } else if constexpr (F == 4) {                                 // 64 B per row: 8 lanes x 8 B, 8 rows per instruction
        char* b = dst + (cta / 256) * (1024 * RS) + (cta % 256) * 64;
        #pragma unroll 4
        for (int it = 0; it < 32; ++it) {
            const int row = (it * 4 + w) * 8 + (l >> 3);
            *(reinterpret_cast<u2*>(b + (size_t)row * RS) + (l & 7)) = u2{v, v + it};
        }
    } else if constexpr (F == 7) {                                 // 64 B per row: 4 lanes x 16 B, 16 rows per instruction
        char* b = dst + (cta / 256) * (1024 * RS) + (cta % 256) * 64;
        #pragma unroll 4
        for (int it = 0; it < 16; ++it) {
            const int row = (it * 4 + w) * 16 + (l >> 2);
            __builtin_nontemporal_store(u4{v, v + it, v ^ 3u, v + 7u}, reinterpret_cast<u4*>(b + (size_t)row * RS) + (l & 3));
        }
    } else if constexpr (F == 9 || F == 10) {
        // CTA owns 512 rows x 128 B (64 KB) of a [T, 64, 64] bf16 tensor (row stride 8 KB): 8 chunk-rows blocks
        char* b = dst + (cta / 64) * (512 * 8192) + (cta % 64) * 128;
        if constexpr (F == 9) {
            #pragma unroll 4
            for (int it = 0; it < 8; ++it)                          // 8 x (16 rows per wave x 4 waves) = 512 rows
                #pragma unroll
                for (int cb = 0; cb < 4; ++cb) {
                    const int row = (it * 4 + w) * 16 + (l & 15);
                    *(reinterpret_cast<u2*>(b + (size_t)row * 8192 + 32 * cb) + (l >> 4)) = u2{v, v + it + cb};
                }
        } else {
            #pragma unroll 4
            for (int it = 0; it < 16; ++it) {                       // 16 x (8 rows x 4 waves) x ... 8 rows per instruction
                const int row = (it * 4 + w) * 8 + (l >> 3);
                *(reinterpret_cast<u4*>(b + (size_t)row * 8192) + (l & 7)) = u4{v, v + it, v ^ 3u, v + 7u};
            }
        }
    } else if constexpr (F == 6) {                                 // 2 B per lane: 4 rows (stride 256 B) per lane group
        char* b = dst + cta * PER_CTA;
        #pragma unroll 4
        for (int it = 0; it < 128; ++it) {
            const size_t o = ((size_t)(it * 4 + w) * 64 + l) * 2;  // 128 B per instruction, every lane a 2 B store
            *reinterpret_cast<unsigned short*>(b + ((o / 128) * 128 + ((o % 128) / 32) * 32 + (o % 32))) = (unsigned short)(v + it);
        }
    }
}

template <int F>
static float run(char* d, int lds, int reps) {
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    for (int i = 0; i < 2; ++i) hipLaunchKernelGGL(k_st<F>, dim3(CTAS), dim3(256), lds, 0, d, 1u + i);
    CK(hipDeviceSynchronize());
    float best = 1e30f;
    for (int r = 0; r < reps; ++r) {
        CK(hipEventRecord(a));
        hipLaunchKernelGGL(k_st<F>, dim3(CTAS), dim3(256), lds, 0, d, 17u + r);
        CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
        float ms; CK(hipEventElapsedTime(&ms, a, b)); if (ms < best) best = ms;
    }
    return best;
}

int main(int argc, char** argv) {
    const size_t span = (size_t)CTAS / 64 * 256 * RS + RS * 1024;   // covers the strided forms
    char* d; CK(hipMalloc(&d, span > CTAS * PER_CTA ? span : CTAS * PER_CTA));
    std::vector<int> forms; for (int i = 1; i < argc; ++i) forms.push_back(atoi(argv[i]));
    if (forms.empty()) forms = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    const char* name[] = {"16B contiguous", "8B contiguous", "4B contiguous", "16B, 256B row segments", "8B, 64B row segments",
                          "16B contiguous, nt", "2B scattered", "16B, 64B row segments, nt", "16B, 256B row segments, nt", "8B, 32B row segs x4 instr", "16B, 128B row segments"};
    const int ldsv[] = {0, 21504, 32768};                            // residency: 0 = occupancy-limited by waves, 3 / 2 CTAs per CU
    const double bytes = (double)CTAS * PER_CTA;
    printf("useful bytes per launch %.0f MB, CTAs %d x 256\n", bytes / 1e6, CTAS);
    for (int f : forms) for (int li = 0; li < 3; ++li) {
        float ms = 0;
        switch (f) {
            case 0: ms = run<0>(d, ldsv[li], 8); break; case 1: ms = run<1>(d, ldsv[li], 8); break;
            case 2: ms = run<2>(d, ldsv[li], 8); break; case 3: ms = run<3>(d, ldsv[li], 8); break;
            case 4: ms = run<4>(d, ldsv[li], 8); break; case 5: ms = run<5>(d, ldsv[li], 8); break;
            case 6: ms = run<6>(d, ldsv[li], 8); break; case 7: ms = run<7>(d, ldsv[li], 8); break;
            case 8: ms = run<8>(d, ldsv[li], 8); break; case 9: ms = run<9>(d, ldsv[li], 8); break;
            default: ms = run<10>(d, ldsv[li], 8); break;
        }
        printf("form %d %-28s lds %5d B: %8.1f us  %7.1f GB/s\n", f, name[f], ldsv[li], ms * 1e3, bytes / (ms * 1e6));
    }
    return 0;
}
