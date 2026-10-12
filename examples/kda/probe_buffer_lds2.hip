// gfx936 probe 2 for the pipelined s4b (round 12): the primitives the kernel relies on.
//  T1  buffer_load_dwordx4 ... lds with an out-of-range voffset deposits ZEROS in LDS (row masking)
//  T2  buffer_store_dwordx4 with an out-of-range voffset is discarded (varlen store masking)
//  T3  "m0" clobber accepted; two stages in flight, manual s_waitcnt vmcnt(N) + barrier publishes
//      the older stage while the newer one is still outstanding (values checked per stage)
// hipcc -O3 --offload-arch=gfx936 probe_buffer_lds2.hip -o probe_buffer_lds2 && ./probe_buffer_lds2
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>

typedef int v4i __attribute__((ext_vector_type(4)));
typedef float f4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ v4i make_srd(const void* base, uint32_t bytes) {
    const uint64_t a = reinterpret_cast<uint64_t>(base);
    v4i r;
    r.x = __builtin_amdgcn_readfirstlane((int)(uint32_t)a);
    r.y = __builtin_amdgcn_readfirstlane((int)(uint32_t)(a >> 32));
    r.z = (int)bytes;
    r.w = (int)0x00020000;
    return r;
}
__device__ __forceinline__ void lds_load16(v4i srd, uint32_t voff, const void* lds_wave_base) {
    const uint32_t m0 = __builtin_amdgcn_readfirstlane((uint32_t)(uintptr_t)lds_wave_base);
    asm volatile("s_mov_b32 m0, %0\n\tbuffer_load_dwordx4 %1, %2, 0 offen offset:0 lds"
                 :: "s"(m0), "v"(voff), "s"(srd) : "memory", "m0");
}
__device__ __forceinline__ void store16(v4i srd, uint32_t voff, f4 v) {
    asm volatile("buffer_store_dwordx4 %0, %1, %2, 0 offen offset:0"
                 :: "v"(v), "v"(voff), "s"(srd) : "memory");
}

// grid 1 x 256: stage s (0,1) loads 4 KB chunk s of src; lanes with odd row index are masked (OOB)
__global__ void __launch_bounds__(256) probe(const int4* src, float* dst, int* out) {
    __shared__ __align__(16) int4 st[2][256];
    const int tid = threadIdx.x, wv = tid >> 6;
    const v4i srd = make_srd(src, 2 * 256 * 16);
    for (int s = 0; s < 2; ++s) {
        const bool masked = (tid & 1);
        const uint32_t voff = masked ? 0xFFFFFFF0u : (uint32_t)((s * 256 + tid) * 16);
        lds_load16(srd, voff, &st[s][wv * 64]);
    }
    // stage 0 complete while stage 1 (4 KB) may still be in flight: 1 op issued after it per wave
    asm volatile("s_waitcnt vmcnt(1)" ::: "memory");
    __syncthreads();
    const int4 a = st[0][tid];
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __syncthreads();
    const int4 b = st[1][tid];
    out[tid * 8 + 0] = a.x; out[tid * 8 + 1] = a.y; out[tid * 8 + 2] = a.z; out[tid * 8 + 3] = a.w;
    out[tid * 8 + 4] = b.x; out[tid * 8 + 5] = b.y; out[tid * 8 + 6] = b.z; out[tid * 8 + 7] = b.w;
    // T2: lanes >= 128 store out of range (discarded); dst holds 128 x 4 floats
    const v4i dsrd = make_srd(dst, 128 * 16);
    const uint32_t so = tid < 128 ? (uint32_t)(tid * 16) : (uint32_t)(4096 + tid * 16);
    store16(dsrd, so, f4{(float)tid, 1.f, 2.f, 3.f});
}

int main() {
    std::vector<int4> h(512);
    for (int i = 0; i < 512; ++i) h[i] = int4{i + 1, 3 * i + 7, -i - 1, i ^ 0x5a5a};
    int4* d_src; float* d_dst; int* d_out;
    hipMalloc(&d_src, 512 * 16); hipMalloc(&d_dst, 256 * 16); hipMalloc(&d_out, 256 * 8 * 4);
    hipMemcpy(d_src, h.data(), 512 * 16, hipMemcpyHostToDevice);
    std::vector<float> sentinel(256 * 4, -7.f);
    hipMemcpy(d_dst, sentinel.data(), 256 * 16, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(probe, dim3(1), dim3(256), 0, 0, d_src, d_dst, d_out);
    const hipError_t e = hipDeviceSynchronize();
    std::vector<int> o(256 * 8); std::vector<float> d(256 * 4);
    hipMemcpy(o.data(), d_out, o.size() * 4, hipMemcpyDeviceToHost);
    hipMemcpy(d.data(), d_dst, d.size() * 4, hipMemcpyDeviceToHost);
    long bad1 = 0, bad3 = 0, bad2 = 0;
    for (int t = 0; t < 256; ++t)
        for (int s = 0; s < 2; ++s) {
            const int* v = &o[t * 8 + 4 * s];
            if (t & 1) bad1 += (v[0] | v[1] | v[2] | v[3]) != 0;                        // OOB -> zeros
            else {
                const int4 r = h[s * 256 + t];
                bad3 += v[0] != r.x || v[1] != r.y || v[2] != r.z || v[3] != r.w;
            }
        }
    for (int t = 0; t < 256; ++t) {
        if (t < 128) bad2 += d[t * 4] != (float)t || d[t * 4 + 1] != 1.f;
        else bad2 += d[t * 4] != -7.f;                                                   // untouched
    }
    printf("err=%s\nT1 OOB lanes deposit zeros: %s (%ld bad)\n", hipGetErrorString(e), bad1 ? "FAIL" : "ok", bad1);
    printf("T3 two stages + vmcnt(1) publish, values: %s (%ld bad)\n", bad3 ? "FAIL" : "ok", bad3);
    printf("T2 OOB buffer_store discarded / in-range written: %s (%ld bad)\n", bad2 ? "FAIL" : "ok", bad2);
    return (bad1 | bad2 | bad3) != 0;
}
