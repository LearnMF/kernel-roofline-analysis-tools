// VMEM completion latency seen by s_waitcnt vmcnt(0): one 8 B/lane store (or load) per iteration, 1 wave / CU
#include <hip/hip_runtime.h>
#include <cstdio>
typedef unsigned u2 __attribute__((ext_vector_type(2)));
template <int ST>
__global__ void __launch_bounds__(64) k(char* g, int n, size_t mask, unsigned* out) {
    unsigned acc = threadIdx.x;
    const size_t base = (size_t)blockIdx.x * 1048576;
    for (int it = 0; it < n; ++it) {
        char* p = g + ((base + (size_t)it * 4096) & mask) + threadIdx.x * 8;
        if (ST) { *reinterpret_cast<u2*>(p) = u2{acc, (unsigned)it}; asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); }
        else { u2 v = *reinterpret_cast<const u2*>(p); asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); acc ^= v.x; }
    }
    if (acc == 0x9e3779b9u) out[threadIdx.x] = acc;
}
int main() {
    char* g; unsigned* o; const size_t big = (size_t)1 << 30;
    hipMalloc(&g, big); hipMalloc(&o, 4096); hipMemset(g, 0, big);
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    for (int st = 0; st < 2; ++st) for (size_t ws : {(size_t)4 << 20, big}) {
        const int n = 4000;
        auto L = [&] { if (st) hipLaunchKernelGGL(k<1>, dim3(80), dim3(64), 0, 0, g, n, ws - 1, o); else hipLaunchKernelGGL(k<0>, dim3(80), dim3(64), 0, 0, g, n, ws - 1, o); };
        L(); hipDeviceSynchronize(); hipEventRecord(a); L(); hipEventRecord(b); hipEventSynchronize(b);
        float ms; hipEventElapsedTime(&ms, a, b);
        printf("VMEMCOMPL %s ws=%zuMB issue_to_vmcnt0_cycles=%.0f\n", st ? "store_dwordx2" : "load_dwordx2", ws >> 20, ms * 1e-3 * 1.5e9 / n);
    }
}
