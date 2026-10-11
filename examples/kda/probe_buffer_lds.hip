// gfx936 feasibility probe for the direct global->LDS path (DCU textbook 6.5.4: "buffer_load_dwordx4 ...
// lds", m0 = the wave's LDS base).  Each wave copies 64 x 16 B from global into its LDS slice with ONE
// buffer_load_dwordx4 ... lds, then reads it back with ds_read and writes it out.  Two variants:
//   A: the clang builtin __builtin_amdgcn_raw_buffer_load_lds (if the toolchain has it)
//   B: inline asm with a hand-built buffer resource (V#) and s_mov_b32 m0
// hipcc -O3 --offload-arch=gfx936 probe_buffer_lds.hip -o probe_buffer_lds && ./probe_buffer_lds
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>

typedef int v4i __attribute__((ext_vector_type(4)));

__device__ __forceinline__ v4i make_srd(const void* base, uint32_t bytes) {
    const uint64_t a = reinterpret_cast<uint64_t>(base);
    v4i r;
    r.x = (int)(uint32_t)a;
    r.y = (int)(uint32_t)(a >> 32);      // stride 0
    r.z = (int)bytes;                    // num_records (bytes, raw buffer)
    r.w = (int)0x00020000;               // dword3 (gfx9-style raw buffer config)
    return r;
}

__global__ void __launch_bounds__(256) copy_asm(const int4* src, int4* dst, int n16) {
    __shared__ __align__(16) int4 lds[256];
    const int tid = threadIdx.x, wv = tid >> 6, l = tid & 63;
    const v4i srd = make_srd(src, n16 * 16);
    const uint32_t m0 = (uint32_t)(uintptr_t)(&lds[wv * 64]);   // wave's LDS base (bytes)
    const uint32_t voff = (uint32_t)((blockIdx.x * 256 + tid) * 16);
    asm volatile(
        "s_mov_b32 m0, %0\n\t"
        "buffer_load_dwordx4 %1, %2, 0 offen offset:0 lds\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        :: "s"(__builtin_amdgcn_readfirstlane(m0)), "v"(voff), "s"(srd) : "memory");
    __syncthreads();
    dst[blockIdx.x * 256 + tid] = lds[tid];
}

#if __has_builtin(__builtin_amdgcn_raw_buffer_load_lds)
#define HAVE_BUILTIN 1
#else
#define HAVE_BUILTIN 0
#endif

int main() {
    const int n16 = 256 * 64;
    std::vector<int4> h(n16), o(n16);
    for (int i = 0; i < n16; ++i) h[i] = int4{i, 3 * i + 1, -i, i ^ 0x5a5a};
    int4 *d_src, *d_dst;
    hipMalloc(&d_src, n16 * 16); hipMalloc(&d_dst, n16 * 16);
    hipMemcpy(d_src, h.data(), n16 * 16, hipMemcpyHostToDevice);
    hipMemset(d_dst, 0, n16 * 16);
    hipLaunchKernelGGL(copy_asm, dim3(n16 / 256), dim3(256), 0, 0, d_src, d_dst, n16);
    const hipError_t e = hipDeviceSynchronize();
    hipMemcpy(o.data(), d_dst, n16 * 16, hipMemcpyDeviceToHost);
    long bad = 0;
    for (int i = 0; i < n16; ++i)
        bad += o[i].x != h[i].x || o[i].y != h[i].y || o[i].z != h[i].z || o[i].w != h[i].w;
    printf("builtin __builtin_amdgcn_raw_buffer_load_lds available: %d\n", HAVE_BUILTIN);
    printf("inline-asm buffer_load_dwordx4 ... lds: err=%s mismatches=%ld / %d\n", hipGetErrorString(e), bad, n16);
    return bad != 0;
}
