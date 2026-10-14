// MMAC dependency latencies without loop-boundary hazards: 16 dependent MMACs per (unrolled) iteration
//   SRCC : acc = mmac(a, b, acc)            (accumulate chain, srcC forwarding)
//   VALU : acc = mmac(a, b, acc); x = acc.x + x; a.x ^= x  (result read by VALU, fed back into srcA)
//   SRCA : acc2 = mmac(cvt(acc), b, acc2)  (result used as an A operand after a pack)
#include <hip/hip_runtime.h>
#include <cstdio>
typedef short v4bh __attribute__((ext_vector_type(4)));
typedef float f4 __attribute__((ext_vector_type(4)));
template <int K>
__global__ void __launch_bounds__(64) k(int n, float* out) {
    f4 acc = {0.f, 0.f, 0.f, (float)threadIdx.x};
    v4bh a = {(short)threadIdx.x, 1, 2, 3}, b = {3, 2, 1, 0};
    float x = 1.f;
    for (int it = 0; it < n; ++it) {
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            acc = __builtin_amdgcn_mmac_f32_16x16x16bf16(a, b, acc);
            if (K == 1) { x += acc[0]; a[0] = (short)__float_as_int(x); }
        }
    }
    if (acc[0] + x == 1.2345f) out[threadIdx.x] = acc[1];
}
int main() {
    float* o; hipMalloc(&o, 4096);
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    for (int kk = 0; kk < 2; ++kk) {
        const int n = 20000;
        auto L = [&] { if (kk == 0) hipLaunchKernelGGL(k<0>, dim3(80), dim3(64), 0, 0, n, o); else hipLaunchKernelGGL(k<1>, dim3(80), dim3(64), 0, 0, n, o); };
        L(); hipDeviceSynchronize();
        hipEventRecord(e0); L(); hipEventRecord(e1); hipEventSynchronize(e1);
        float ms; hipEventElapsedTime(&ms, e0, e1);
        printf("MMACDEP %s cycles_per_dependent_mmac=%.2f\n", kk ? "result->VALU->srcA" : "srcC_chain_unrolled", ms * 1e-3 * 1.5e9 / (n * 16.0));
    }
}
