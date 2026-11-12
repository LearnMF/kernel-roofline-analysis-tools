// LDS bank conflicts of ds_read_m32x16_b16 (and ds_read_b128 / b64 on the same per-lane addresses) on gfx938.
// One dispatch per (op, pattern, reps); hipprof --pmc SQ_LDS_BANK_CONFLICT / SQ_LDS_IDX_ACTIVE / SQ_INSTS_LDS per dispatch.
// Conflict cycles per instruction = (count(reps = R) - count(reps = 0)) / (CTAs * R).
// Patterns (byte offsets per lane) come from lds_m_pat.txt: one line per pattern, 64 integers (16 B aligned, < 32 KB).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

typedef short v8s __attribute__((ext_vector_type(8)));
typedef short v4s __attribute__((ext_vector_type(4)));

template <int OP>
__global__ void __launch_bounds__(64) probe(const int* tab, int pat, int reps, int* out) {
    __shared__ __align__(16) short lds[16384];   // 32 KB
    for (int i = threadIdx.x; i < 16384; i += 64) lds[i] = (short)i;
    __syncthreads();
    const int l = threadIdx.x;
    int off = tab[pat * 64 + l] >> 1;           // shorts
    int acc = 0;
    for (int r = 0; r < reps; ++r) {
        asm volatile("" : "+v"(off));
        if constexpr (OP == 0) {
            const v8s x = __builtin_hcu_ds_read_m32x16_bf16(lds + off);
            acc += x[0] + x[7];
        } else if constexpr (OP == 1) {
            const v8s x = *reinterpret_cast<const v8s*>(lds + off);
            acc += x[0] + x[7];
        } else {
            const v4s x = *reinterpret_cast<const v4s*>(lds + off);
            acc += x[0] + x[3];
        }
    }
    out[blockIdx.x * 64 + l] = acc;
}

int main(int argc, char** argv) {
    const char* fn = argc > 1 ? argv[1] : "lds_m_pat.txt";
    const int R = argc > 2 ? atoi(argv[2]) : 64, CTAS = 72;
    std::vector<int> h;
    std::ifstream f(fn);
    std::string line;
    int np = 0;
    while (std::getline(f, line)) {
        std::istringstream is(line);
        int v, n = 0;
        while (is >> v) { h.push_back(v); ++n; }
        if (n == 64) ++np; else if (n) { fprintf(stderr, "bad line %d (%d values)\n", np, n); return 1; }
    }
    int *tab, *out;
    hipMalloc(&tab, h.size() * 4);
    hipMalloc(&out, CTAS * 64 * 4);
    hipMemcpy(tab, h.data(), h.size() * 4, hipMemcpyHostToDevice);
    // dispatch order: for each pattern, for op 0..2: reps 0 then reps R (printed so the PMC rows can be matched)
    for (int p = 0; p < np; ++p)
        for (int op = 0; op < 3; ++op)
            for (int rr = 0; rr < 2; ++rr) {
                const int reps = rr ? R : 0;
                if (op == 0) hipLaunchKernelGGL(probe<0>, dim3(CTAS), dim3(64), 0, 0, tab, p, reps, out);
                if (op == 1) hipLaunchKernelGGL(probe<1>, dim3(CTAS), dim3(64), 0, 0, tab, p, reps, out);
                if (op == 2) hipLaunchKernelGGL(probe<2>, dim3(CTAS), dim3(64), 0, 0, tab, p, reps, out);
                hipDeviceSynchronize();
                printf("D pat %d op %d reps %d\n", p, op, reps);
            }
    printf("patterns %d R %d CTAs %d\n", np, R, CTAS);
    return 0;
}
