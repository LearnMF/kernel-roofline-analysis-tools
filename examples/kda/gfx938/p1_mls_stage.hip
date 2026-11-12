#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#ifndef MINOUT
#define MINOUT 0
#endif
typedef short v4bh __attribute__((ext_vector_type(4)));
typedef float f4 __attribute__((ext_vector_type(4)));
typedef int v4i __attribute__((ext_vector_type(4)));
constexpr int RS = 8192;            // row stride (elements): H = 64 heads x 128
constexpr int kPadJ2 = 68, kPadA = 68;
__device__ __forceinline__ f4 mmac(v4bh a, v4bh b, f4 c) { return __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(a, b, c, false, false); }
__device__ __forceinline__ short bf_rne(float x) { uint32_t u = __float_as_uint(x); u += 0x7fffu + ((u >> 16) & 1u); return (short)(u >> 16); }
__device__ __forceinline__ float bf2f(short s) { return __uint_as_float(((uint32_t)(unsigned short)s) << 16); }
__device__ __forceinline__ int swz(int r, int j) { return ((((j >> 2) ^ (r >> 3)) & 15) << 2) | (j & 3); }
__device__ __forceinline__ size_t tile_base(int c) { return (size_t)(c >> 6) * 64 * RS + (size_t)(c & 63) * 128; }
// C-fragment element e of lane l -> (row, col) inside the 16x16 tile (same convention for both variants)
__device__ __forceinline__ int cr(int l, int e) { return (l >> 4) + 4 * e; }
__device__ __forceinline__ int cc(int l) { return l & 15; }

__device__ __forceinline__ void load_common(const short* A, const float* beta, int c, short* sA, float* sB) {
  const int tid = threadIdx.x;
  if (tid < 64) sB[tid] = beta[(size_t)c * 64 + tid];
  for (int it = 0; it < 4; ++it) { const int t = tid + 256 * it, r = t >> 4, c4 = (t & 15) << 2;
    *reinterpret_cast<v4bh*>(sA + r * kPadA + c4) = *reinterpret_cast<const v4bh*>(A + (size_t)c * 4096 + r * 64 + c4); }
}

// A: the gfx936 data path (wu u phase): row loads -> VGPR scale/round -> transposed b16 LDS scatter -> b64 fragment reads
__global__ void __launch_bounds__(256) kA(const short* V, const float* beta, const short* A, short* U) {
  __shared__ __align__(16) short slab[128 * kPadJ2];
  __shared__ __align__(16) short sA[64 * kPadA];
  __shared__ float sB[64];
  const int c = blockIdx.x, tid = threadIdx.x, wv = tid >> 6, l = tid & 63;
  load_common(A, beta, c, sA, sB);
  __syncthreads();
  const size_t base = tile_base(c);
  #pragma unroll
  for (int it = 0; it < 8; ++it) {
    const int t = tid + 256 * it, j = t >> 5, vq = (t & 31) << 2;
    const v4bh vv = *reinterpret_cast<const v4bh*>(V + base + (size_t)j * RS + vq);
    const float b = sB[j];
    #pragma unroll
    for (int dv = 0; dv < 4; ++dv) slab[(vq + dv) * kPadJ2 + swz(vq + dv, j)] = bf_rne(bf2f(vv[dv]) * b);
  }
  __syncthreads();
  f4 acc[8];
  #pragma unroll
  for (int vt = 0; vt < 8; ++vt) acc[vt] = f4{0, 0, 0, 0};
  for (int js = 0; js < 4; ++js) {
    const v4bh af = *reinterpret_cast<const v4bh*>(sA + (16 * wv + (l & 15)) * kPadA + 16 * js + (l >> 4) * 4);
    #pragma unroll
    for (int vt = 0; vt < 8; ++vt) {
      const int br = 16 * vt + (l & 15);
      acc[vt] = mmac(af, *reinterpret_cast<const v4bh*>(slab + br * kPadJ2 + swz(br, 16 * js + (l >> 4) * 4)), acc[vt]);
    }
  }
#if MINOUT
  float sum = 0; for (int vt = 0; vt < 8; ++vt) for (int e = 0; e < 4; ++e) sum += acc[vt][e];
  reinterpret_cast<float*>(U)[(size_t)c * 256 + tid] = sum;
#else
  #pragma unroll
  for (int vt = 0; vt < 8; ++vt)
    #pragma unroll
    for (int e = 0; e < 4; ++e) U[(size_t)c * 8192 + (16 * wv + cr(l, e)) * 128 + 16 * vt + cc(l)] = bf_rne(acc[vt][e]);
#endif
}

// C: MLS-native: matrix_load HBM -> LDS (row-major 32x32 tiles, no VGPR) -> ds_read_matrix_format = B fragments
// (k = t) -> in-register scale/round -> MMAC.  Wave w owns columns 32w..32w+31 (each element scaled once).
__global__ void __launch_bounds__(256) kC(const short* V, const float* beta, const short* A, short* U) {
  __shared__ __align__(16) short sV[8 * 1024];
  __shared__ __align__(16) short sA[64 * kPadA];
  __shared__ float sB[64];
  const int c = blockIdx.x, tid = threadIdx.x, w = tid >> 6, l = tid & 63;
  const size_t base = tile_base(c);
  #pragma unroll
  for (int q2 = 0; q2 < 2; ++q2) {            // this wave's two 32x32 tiles: row half th, column block w
    const int th = q2;
    uint64_t a = (uint64_t)(V + base + (size_t)(32 * th) * RS + 32 * w) & 0xffffffffffffull;
    v4i r; r[0] = __builtin_amdgcn_readfirstlane((int)(uint32_t)a); r[1] = __builtin_amdgcn_readfirstlane((int)(uint32_t)(a >> 32)); r[2] = RS; r[3] = 0;
    const int soff = __builtin_amdgcn_readfirstlane((int)(size_t)(sV + (th * 4 + w) * 1024));
    __builtin_hcu_matrix_load_32x32_b16(r, (__attribute__((address_space(3))) short*)(soff), 0, 0, 0, 0, 0);
  }
  load_common(A, beta, c, sA, sB);
  __builtin_amdgcn_s_waitcnt(0);
  __syncthreads();
  f4 acc[4][2];
  #pragma unroll
  for (int mt = 0; mt < 4; ++mt) acc[mt][0] = acc[mt][1] = f4{0, 0, 0, 0};
  const int g = l >> 4;
  for (int js = 0; js < 4; ++js) {
    v4i rr;
    const int m0 = __builtin_amdgcn_readfirstlane((int)(size_t)(sV + ((js >> 1) * 4 + w) * 1024 + (js & 1) * 512));
    asm volatile("s_mov_b32 m0, %1\n\ts_nop 0\n\tds_read_matrix_format %0, m0 offset:0 element:0x2 row:0x2 col:0x1 alt:0x0\n\ts_waitcnt lgkmcnt(0)"
                 : "=v"(rr) : "s"(m0));
    const short* rv = reinterpret_cast<const short*>(&rr);
    const f4 bb = *reinterpret_cast<const f4*>(sB + 16 * js + 4 * g);
    v4bh b0, b1;
    #pragma unroll
    for (int i = 0; i < 4; ++i) { b0[i] = bf_rne(bf2f(rv[i]) * bb[i]); b1[i] = bf_rne(bf2f(rv[4 + i]) * bb[i]); }
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
      const v4bh af = *reinterpret_cast<const v4bh*>(sA + (16 * mt + (l & 15)) * kPadA + 16 * js + (l >> 4) * 4);
      acc[mt][0] = mmac(af, b0, acc[mt][0]);
      acc[mt][1] = mmac(af, b1, acc[mt][1]);
    }
  }
#if MINOUT
  float sum = 0; for (int mt = 0; mt < 4; ++mt) for (int q = 0; q < 2; ++q) for (int e = 0; e < 4; ++e) sum += acc[mt][q][e];
  reinterpret_cast<float*>(U)[(size_t)c * 256 + tid] = sum;
#else
  #pragma unroll
  for (int mt = 0; mt < 4; ++mt)
    #pragma unroll
    for (int q = 0; q < 2; ++q)
      #pragma unroll
      for (int e = 0; e < 4; ++e) U[(size_t)c * 8192 + (16 * mt + cr(l, e)) * 128 + 32 * w + 16 * q + cc(l)] = bf_rne(acc[mt][q][e]);
#endif
}

// C2: C with the four fragment reads issued back to back, one wait: matrix_load HBM -> LDS (row-major 32x32 tiles, no VGPR) -> ds_read_matrix_format = B fragments
// (k = t) -> in-register scale/round -> MMAC.  Wave w owns columns 32w..32w+31 (each element scaled once).
__global__ void __launch_bounds__(256) kC2(const short* V, const float* beta, const short* A, short* U) {
  __shared__ __align__(16) short sV[8 * 1024];
  __shared__ __align__(16) short sA[64 * kPadA];
  __shared__ float sB[64];
  const int c = blockIdx.x, tid = threadIdx.x, w = tid >> 6, l = tid & 63;
  const size_t base = tile_base(c);
  #pragma unroll
  for (int q2 = 0; q2 < 2; ++q2) {            // this wave's two 32x32 tiles: row half th, column block w
    const int th = q2;
    uint64_t a = (uint64_t)(V + base + (size_t)(32 * th) * RS + 32 * w) & 0xffffffffffffull;
    v4i r; r[0] = __builtin_amdgcn_readfirstlane((int)(uint32_t)a); r[1] = __builtin_amdgcn_readfirstlane((int)(uint32_t)(a >> 32)); r[2] = RS; r[3] = 0;
    const int soff = __builtin_amdgcn_readfirstlane((int)(size_t)(sV + (th * 4 + w) * 1024));
    __builtin_hcu_matrix_load_32x32_b16(r, (__attribute__((address_space(3))) short*)(soff), 0, 0, 0, 0, 0);
  }
  load_common(A, beta, c, sA, sB);
  __builtin_amdgcn_s_waitcnt(0);
  __syncthreads();
  f4 acc[4][2];
  #pragma unroll
  for (int mt = 0; mt < 4; ++mt) acc[mt][0] = acc[mt][1] = f4{0, 0, 0, 0};
  const int g = l >> 4;
  v4i rr[4];
  #pragma unroll
  for (int js = 0; js < 4; ++js) {
    const int m0 = __builtin_amdgcn_readfirstlane((int)(size_t)(sV + ((js >> 1) * 4 + w) * 1024 + (js & 1) * 512));
    asm volatile("s_mov_b32 m0, %1\n\ts_nop 0\n\tds_read_matrix_format %0, m0 offset:0 element:0x2 row:0x2 col:0x1 alt:0x0" : "=v"(rr[js]) : "s"(m0));
  }
  asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(rr[0]), "+v"(rr[1]), "+v"(rr[2]), "+v"(rr[3]) :: "memory");
  #pragma unroll
  for (int js = 0; js < 4; ++js) {
    const short* rv = reinterpret_cast<const short*>(&rr[js]);
    const f4 bb = *reinterpret_cast<const f4*>(sB + 16 * js + 4 * g);
    v4bh b0, b1;
    #pragma unroll
    for (int i = 0; i < 4; ++i) { b0[i] = bf_rne(bf2f(rv[i]) * bb[i]); b1[i] = bf_rne(bf2f(rv[4 + i]) * bb[i]); }
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
      const v4bh af = *reinterpret_cast<const v4bh*>(sA + (16 * mt + (l & 15)) * kPadA + 16 * js + (l >> 4) * 4);
      acc[mt][0] = mmac(af, b0, acc[mt][0]);
      acc[mt][1] = mmac(af, b1, acc[mt][1]);
    }
  }
#if MINOUT
  float sum = 0; for (int mt = 0; mt < 4; ++mt) for (int q = 0; q < 2; ++q) for (int e = 0; e < 4; ++e) sum += acc[mt][q][e];
  reinterpret_cast<float*>(U)[(size_t)c * 256 + tid] = sum;
#else
  #pragma unroll
  for (int mt = 0; mt < 4; ++mt)
    #pragma unroll
    for (int q = 0; q < 2; ++q)
      #pragma unroll
      for (int e = 0; e < 4; ++e) U[(size_t)c * 8192 + (16 * mt + cr(l, e)) * 128 + 32 * w + 16 * q + cc(l)] = bf_rne(acc[mt][q][e]);
#endif
}

// C4: C2 + A (64x64) through MLS (swizzled) and ds_read_matrix_trans_format: no VGPR staging, no per-lane addresses: matrix_load HBM -> LDS (row-major 32x32 tiles, no VGPR) -> ds_read_matrix_format = B fragments
// (k = t) -> in-register scale/round -> MMAC.  Wave w owns columns 32w..32w+31 (each element scaled once).
__global__ void __launch_bounds__(256) kC4(const short* V, const float* beta, const short* A, short* U) {
  __shared__ __align__(16) short sV[8 * 1024];
  __shared__ __align__(16) short sAm[4 * 1024];
  __shared__ float sB[64];
  const int c = blockIdx.x, tid = threadIdx.x, w = tid >> 6, l = tid & 63;
  const size_t base = tile_base(c);
  #pragma unroll
  for (int q2 = 0; q2 < 2; ++q2) {            // this wave's two 32x32 tiles: row half th, column block w
    const int th = q2;
    uint64_t a = (uint64_t)(V + base + (size_t)(32 * th) * RS + 32 * w) & 0xffffffffffffull;
    v4i r; r[0] = __builtin_amdgcn_readfirstlane((int)(uint32_t)a); r[1] = __builtin_amdgcn_readfirstlane((int)(uint32_t)(a >> 32)); r[2] = RS; r[3] = 0;
    const int soff = __builtin_amdgcn_readfirstlane((int)(size_t)(sV + (th * 4 + w) * 1024));
    __builtin_hcu_matrix_load_32x32_b16(r, (__attribute__((address_space(3))) short*)(soff), 0, 0, 0, 0, 0);
  }
  {   // wave w: A tile w = rows 32 (w >> 1).., cols 32 (w & 1)..; swizzled LDS layout for the trans reads
    uint64_t a = (uint64_t)(A + (size_t)c * 4096 + (size_t)(32 * (w >> 1)) * 64 + 32 * (w & 1)) & 0xffffffffffffull;
    v4i r; r[0] = __builtin_amdgcn_readfirstlane((int)(uint32_t)a); r[1] = __builtin_amdgcn_readfirstlane((int)(uint32_t)(a >> 32)); r[2] = 64; r[3] = 0;
    const int soff = __builtin_amdgcn_readfirstlane((int)(size_t)(sAm + w * 1024)) + (int)0x80000000;
    __builtin_hcu_matrix_load_32x32_b16(r, (__attribute__((address_space(3))) short*)(soff), 0, 0, 0, 0, 0);
  }
  if (tid < 64) sB[tid] = beta[(size_t)c * 64 + tid];
  __builtin_amdgcn_s_waitcnt(0);
  __syncthreads();
  f4 acc[4][2];
  #pragma unroll
  for (int mt = 0; mt < 4; ++mt) acc[mt][0] = acc[mt][1] = f4{0, 0, 0, 0};
  const int g = l >> 4;
  v4i rr[4], ra[4][2];
  #pragma unroll
  for (int mt = 0; mt < 4; ++mt)
    #pragma unroll
    for (int jp = 0; jp < 2; ++jp) {
      const int m0 = __builtin_amdgcn_readfirstlane((int)(size_t)(sAm + ((mt >> 1) * 2 + jp) * 1024 + (mt & 1) * 512));
      asm volatile("s_mov_b32 m0, %1\n\ts_nop 0\n\tds_read_matrix_trans_format %0, m0 offset:0 element:0x2 row:0x2 col:0x1 alt:0x0" : "=v"(ra[mt][jp]) : "s"(m0));
    }
  #pragma unroll
  for (int js = 0; js < 4; ++js) {
    const int m0 = __builtin_amdgcn_readfirstlane((int)(size_t)(sV + ((js >> 1) * 4 + w) * 1024 + (js & 1) * 512));
    asm volatile("s_mov_b32 m0, %1\n\ts_nop 0\n\tds_read_matrix_format %0, m0 offset:0 element:0x2 row:0x2 col:0x1 alt:0x0" : "=v"(rr[js]) : "s"(m0));
  }
  asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(rr[0]), "+v"(rr[1]), "+v"(rr[2]), "+v"(rr[3]), "+v"(ra[0][0]), "+v"(ra[0][1]), "+v"(ra[1][0]), "+v"(ra[1][1]), "+v"(ra[2][0]), "+v"(ra[2][1]), "+v"(ra[3][0]), "+v"(ra[3][1]) :: "memory");
  #pragma unroll
  for (int js = 0; js < 4; ++js) {
    const short* rv = reinterpret_cast<const short*>(&rr[js]);
    const f4 bb = *reinterpret_cast<const f4*>(sB + 16 * js + 4 * g);
    v4bh b0, b1;
    #pragma unroll
    for (int i = 0; i < 4; ++i) { b0[i] = bf_rne(bf2f(rv[i]) * bb[i]); b1[i] = bf_rne(bf2f(rv[4 + i]) * bb[i]); }
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
      const v4bh* ap = reinterpret_cast<const v4bh*>(&ra[mt][js >> 1]);
      const v4bh af = ap[js & 1];
      acc[mt][0] = mmac(af, b0, acc[mt][0]);
      acc[mt][1] = mmac(af, b1, acc[mt][1]);
    }
  }
#if MINOUT
  float sum = 0; for (int mt = 0; mt < 4; ++mt) for (int q = 0; q < 2; ++q) for (int e = 0; e < 4; ++e) sum += acc[mt][q][e];
  reinterpret_cast<float*>(U)[(size_t)c * 256 + tid] = sum;
#else
  #pragma unroll
  for (int mt = 0; mt < 4; ++mt)
    #pragma unroll
    for (int q = 0; q < 2; ++q)
      #pragma unroll
      for (int e = 0; e < 4; ++e) U[(size_t)c * 8192 + (16 * mt + cr(l, e)) * 128 + 32 * w + 16 * q + cc(l)] = bf_rne(acc[mt][q][e]);
#endif
}

// C3: persistent CTAs + double-buffered MLS: the next chunk-head's V tiles stream into the other LDS buffer
// while this one computes (zero-copy double pipeline).
__global__ void __launch_bounds__(256) kC3(const short* V, const float* beta, const short* A, short* U, int NC) {
  __shared__ __align__(16) short sV[2][8 * 1024];
  __shared__ __align__(16) short sA[64 * kPadA];
  __shared__ float sB[64];
  const int tid = threadIdx.x, w = tid >> 6, l = tid & 63, g = l >> 4;
  auto issue = [&](int c, int buf) {
    const size_t base = tile_base(c);
    #pragma unroll
    for (int th = 0; th < 2; ++th) {
      uint64_t a = (uint64_t)(V + base + (size_t)(32 * th) * RS + 32 * w) & 0xffffffffffffull;
      v4i r; r[0] = __builtin_amdgcn_readfirstlane((int)(uint32_t)a); r[1] = __builtin_amdgcn_readfirstlane((int)(uint32_t)(a >> 32)); r[2] = RS; r[3] = 0;
      const int soff = __builtin_amdgcn_readfirstlane((int)(size_t)(&sV[buf][(th * 4 + w) * 1024]));
      __builtin_hcu_matrix_load_32x32_b16(r, (__attribute__((address_space(3))) short*)(soff), 0, 0, 0, 0, 0);
    }
  };
  int c = blockIdx.x, buf = 0;
  if (c < NC) issue(c, 0);
  for (; c < NC; c += gridDim.x, buf ^= 1) {
    __syncthreads();                       // previous iteration's sA / sB / sV[buf^1] reads done
    load_common(A, beta, c, sA, sB);
    const int cn = c + gridDim.x;
    __builtin_amdgcn_s_waitcnt(0);         // this chunk's MLS tiles (and sA / sB loads) landed
    __syncthreads();
    if (cn < NC) issue(cn, buf ^ 1);       // prefetch the next chunk-head while computing this one
    f4 acc[4][2];
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt) acc[mt][0] = acc[mt][1] = f4{0, 0, 0, 0};
    v4i rr[4];
    #pragma unroll
    for (int js = 0; js < 4; ++js) {
      const int m0 = __builtin_amdgcn_readfirstlane((int)(size_t)(&sV[buf][((js >> 1) * 4 + w) * 1024 + (js & 1) * 512]));
      asm volatile("s_mov_b32 m0, %1\n\ts_nop 0\n\tds_read_matrix_format %0, m0 offset:0 element:0x2 row:0x2 col:0x1 alt:0x0" : "=v"(rr[js]) : "s"(m0));
    }
    asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(rr[0]), "+v"(rr[1]), "+v"(rr[2]), "+v"(rr[3]) :: "memory");
    #pragma unroll
    for (int js = 0; js < 4; ++js) {
      const short* rv = reinterpret_cast<const short*>(&rr[js]);
      const f4 bb = *reinterpret_cast<const f4*>(sB + 16 * js + 4 * g);
      v4bh b0, b1;
      #pragma unroll
      for (int i = 0; i < 4; ++i) { b0[i] = bf_rne(bf2f(rv[i]) * bb[i]); b1[i] = bf_rne(bf2f(rv[4 + i]) * bb[i]); }
      #pragma unroll
      for (int mt = 0; mt < 4; ++mt) {
        const v4bh af = *reinterpret_cast<const v4bh*>(sA + (16 * mt + (l & 15)) * kPadA + 16 * js + (l >> 4) * 4);
        acc[mt][0] = mmac(af, b0, acc[mt][0]);
        acc[mt][1] = mmac(af, b1, acc[mt][1]);
      }
    }
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt)
      #pragma unroll
      for (int q = 0; q < 2; ++q)
        #pragma unroll
        for (int e = 0; e < 4; ++e) U[(size_t)c * 8192 + (16 * mt + cr(l, e)) * 128 + 32 * w + 16 * q + cc(l)] = bf_rne(acc[mt][q][e]);
  }
}

int main(int argc, char** argv) {
  const int NC = argc > 1 ? atoi(argv[1]) : 8192;      // chunk-heads (8192/64: NT 128 x H 64)
  const size_t nV = (size_t)(NC / 64) * 64 * RS;
  std::vector<short> hV(nV), hA((size_t)NC * 4096); std::vector<float> hB((size_t)NC * 64);
  srand(7);
  auto rbf = [] { float f = (rand() / (float)RAND_MAX - 0.5f) * 2.f; uint32_t u; memcpy(&u, &f, 4); return (short)(u >> 16); };
  for (auto& x : hV) x = rbf(); for (auto& x : hA) x = rbf(); for (auto& x : hB) x = rand() / (float)RAND_MAX;
  short *dV, *dA, *dUa, *dUc; float* dB;
  hipMalloc(&dV, nV * 2); hipMalloc(&dA, hA.size() * 2); hipMalloc(&dB, hB.size() * 4); hipMalloc(&dUa, (size_t)NC * 8192 * 2); hipMalloc(&dUc, (size_t)NC * 8192 * 2);
  hipMemcpy(dV, hV.data(), nV * 2, hipMemcpyHostToDevice); hipMemcpy(dA, hA.data(), hA.size() * 2, hipMemcpyHostToDevice); hipMemcpy(dB, hB.data(), hB.size() * 4, hipMemcpyHostToDevice);
  hipLaunchKernelGGL(kA, dim3(NC), dim3(256), 0, 0, dV, dB, dA, dUa);
  hipLaunchKernelGGL(kC, dim3(NC), dim3(256), 0, 0, dV, dB, dA, dUc);
  hipDeviceSynchronize();
  short* dUx; hipMalloc(&dUx, (size_t)NC * 8192 * 2);
  std::vector<short> ux((size_t)NC * 8192);
  std::vector<short> ua((size_t)NC * 8192), uc((size_t)NC * 8192);
  hipMemcpy(ua.data(), dUa, ua.size() * 2, hipMemcpyDeviceToHost); hipMemcpy(uc.data(), dUc, uc.size() * 2, hipMemcpyDeviceToHost);
  size_t diff = 0; for (size_t i = 0; i < ua.size(); ++i) diff += ua[i] != uc[i];
  // CPU spot check of variant A semantics (tolerance): U[m][n] = sum_k A[m][k] * bf16(V[k][n] * beta[k])
  double maxrel = 0, maxrel_t = 0;
  for (int c : {0, NC / 2, NC - 1}) for (int m : {0, 17, 63}) for (int n : {0, 33, 127}) {
    double s = 0; const size_t base = (size_t)(c >> 6) * 64 * RS + (size_t)(c & 63) * 128;
    auto b2f = [](short x) { uint32_t u = ((uint32_t)(unsigned short)x) << 16; float f; memcpy(&f, &u, 4); return f; };
    for (int k = 0; k < 64; ++k) { float vb = b2f(hV[base + (size_t)k * RS + n]) * hB[(size_t)c * 64 + k]; uint32_t u; memcpy(&u, &vb, 4); u += 0x7fff + ((u >> 16) & 1); s += (double)b2f(hA[(size_t)c * 4096 + m * 64 + k]) * b2f((short)(u >> 16)); }
    // locate (m, n) under the transposed-in-tile convention too
    const int mt_ = m / 16, nt_ = n / 16, mi = m % 16, ni = n % 16;
    const double got = b2f(ua[(size_t)c * 8192 + m * 128 + n]);
    const double got_t = b2f(ua[(size_t)c * 8192 + (16 * mt_ + ni) * 128 + 16 * nt_ + mi]);
    maxrel = fmax(maxrel, fabs(got - s) / (fabs(s) + 1e-2)); maxrel_t = fmax(maxrel_t, fabs(got_t - s) / (fabs(s) + 1e-2));
  }
  printf("bitwise A vs C: %zu / %zu differ | A vs CPU fp64 max rel %.2e (in-tile transposed convention %.2e)\n", diff, ua.size(), maxrel, maxrel_t);
  const int P = argc > 2 ? atoi(argv[2]) : 72 * 2;
  for (int v = 2; v < 5; ++v) {
    hipMemset(dUx, 0, (size_t)NC * 8192 * 2);
    if (v == 2) hipLaunchKernelGGL(kC2, dim3(NC), dim3(256), 0, 0, dV, dB, dA, dUx); else if (v == 3) hipLaunchKernelGGL(kC3, dim3(P), dim3(256), 0, 0, dV, dB, dA, dUx, NC); else hipLaunchKernelGGL(kC4, dim3(NC), dim3(256), 0, 0, dV, dB, dA, dUx);
    hipDeviceSynchronize(); hipMemcpy(ux.data(), dUx, ux.size() * 2, hipMemcpyDeviceToHost);
    size_t d2 = 0; for (size_t i = 0; i < ua.size(); ++i) d2 += ua[i] != ux[i];
    printf("bitwise A vs %s: %zu differ\n", v == 2 ? "C2" : v == 3 ? "C3" : "C4", d2);
  }
  hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
  for (int v = 0; v < 5; ++v) {
    float best = 1e9;
    for (int rep = 0; rep < 20; ++rep) {
      hipEventRecord(e0);
      if (v == 0) hipLaunchKernelGGL(kA, dim3(NC), dim3(256), 0, 0, dV, dB, dA, dUa); else if (v == 1) hipLaunchKernelGGL(kC, dim3(NC), dim3(256), 0, 0, dV, dB, dA, dUc); else if (v == 2) hipLaunchKernelGGL(kC2, dim3(NC), dim3(256), 0, 0, dV, dB, dA, dUx); else if (v == 3) hipLaunchKernelGGL(kC3, dim3(P), dim3(256), 0, 0, dV, dB, dA, dUx, NC); else hipLaunchKernelGGL(kC4, dim3(NC), dim3(256), 0, 0, dV, dB, dA, dUx);
      hipEventRecord(e1); hipEventSynchronize(e1); float ms; hipEventElapsedTime(&ms, e0, e1); best = fmin(best, ms);
    }
    const double bytes = (double)NC * (8192 * 2 * 2 + 4096 * 2 + 256);
    const char* nm[5] = {"A gfx936-path        ", "C MLS-native         ", "C2 MLS, batched reads", "C3 MLS persistent 2buf", "C4 MLS A+V, no VGPR in"};
    printf("%s: %.1f us (best of 20), %.0f GB/s effective\n", nm[v], best * 1e3, bytes / (best * 1e-3) / 1e9);
  }
}
