// Calibrate PMC semantics on gfx936 for the hierarchical roofline:
//   k_bf16 : N x v_mmac_f32_16x16x16_bf16 per wave   (8192 FLOP each)
//   k_f32  : N x v_mmac_f32_16x16x8_f32  per wave   (4096 FLOP each)
//   k_mix  : N bf16 + N f32 per wave
//   k_lds  : N x ds_read_b128 per lane (16 B/lane, conflict-free)  -> known LDS bytes
#include <hip/hip_runtime.h>
#include <cstdio>
typedef short v4bh __attribute__((ext_vector_type(4)));
typedef float f4 __attribute__((ext_vector_type(4)));
typedef float f2 __attribute__((ext_vector_type(2)));
constexpr int N = 4096;
__global__ void __launch_bounds__(256) k_bf16(float* out) {
    v4bh a = {1, 2, 3, 4}; f4 c = {0, 0, 0, 0};
    for (int i = 0; i < N; ++i) c = __builtin_amdgcn_mmac_f32_16x16x16bf16(a, a, c);
    if (c[0] == 1234.f) out[threadIdx.x] = c[1];
}
__global__ void __launch_bounds__(256) k_f32(float* out) {
    f2 a = {1.f, 2.f}; f4 c = {0, 0, 0, 0};
    for (int i = 0; i < N; ++i) c = __builtin_amdgcn_mmac_f32_16x16x8f32(a, a, c);
    if (c[0] == 1234.f) out[threadIdx.x] = c[1];
}
__global__ void __launch_bounds__(256) k_mix(float* out) {
    v4bh a = {1, 2, 3, 4}; f2 b = {1.f, 2.f}; f4 c = {0, 0, 0, 0}, d = c;
    for (int i = 0; i < N; ++i) { c = __builtin_amdgcn_mmac_f32_16x16x16bf16(a, a, c); d = __builtin_amdgcn_mmac_f32_16x16x8f32(b, b, d); }
    if (c[0] + d[0] == 1234.f) out[threadIdx.x] = c[1] + d[1];
}
__global__ void __launch_bounds__(256) k_lds(float* out) {
    __shared__ __align__(16) float s[256 * 4];
    reinterpret_cast<f4*>(s)[threadIdx.x] = f4{1.f, 2.f, 3.f, (float)threadIdx.x};
    __syncthreads();
    f4 acc = {0, 0, 0, 0};
    for (int i = 0; i < N; ++i) {
        const int idx = (threadIdx.x + i) & 255;
        f4 v; asm volatile("ds_read_b128 %0, %1\n\ts_waitcnt lgkmcnt(0)" : "=v"(v) : "v"(idx * 16) : "memory");
        acc += v;
    }
    if (acc[0] == 1234.f) out[threadIdx.x] = acc[1];
}
int main() {
    float* o; hipMalloc(&o, 4096);
    const int G = 640;                                   // 640 CTAs x 4 waves
    for (int r = 0; r < 2; ++r) {
        hipLaunchKernelGGL(k_bf16, dim3(G), dim3(256), 0, 0, o);
        hipLaunchKernelGGL(k_f32, dim3(G), dim3(256), 0, 0, o);
        hipLaunchKernelGGL(k_mix, dim3(G), dim3(256), 0, 0, o);
        hipLaunchKernelGGL(k_lds, dim3(G), dim3(256), 0, 0, o);
    }
    hipDeviceSynchronize();
    printf("expected per kernel: waves=%d  mmac/kernel=%lld  lds_bytes(k_lds)=%lld\n", G * 4,
           (long long)G * 4 * N, (long long)G * 256 * N * 16);
    return 0;
}
