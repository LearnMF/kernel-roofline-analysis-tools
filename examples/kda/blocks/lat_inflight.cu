// Memory latency and the in-flight bytes needed to reach bandwidth (Little's law), portable CUDA / HIP.
//   (1) device properties; (2) pointer-chase latency (1 thread, random cycle) at working sets inside L2 and in DRAM;
//   (3) streaming-read bandwidth vs bytes in flight per CU/SM: B blocks per CU/SM x 256 threads x U independent 16 B
//       loads per thread per step.
// build: hipcc -O3 --offload-arch=gfx936 -x hip lat_inflight.cu -o lat  |  nvcc -O3 -arch=sm_80 lat_inflight.cu -o lat
#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#define gpu(x) hip##x
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuDeviceProp hipDeviceProp_t
#define SMEM_SM(p) (p).maxSharedMemoryPerMultiProcessor
#define REGS_SM(p) 0
#else
#include <cuda_runtime.h>
#define gpu(x) cuda##x
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuDeviceProp cudaDeviceProp
#define SMEM_SM(p) (p).sharedMemPerMultiprocessor
#define REGS_SM(p) (p).regsPerMultiprocessor
#endif
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <random>

__global__ void chase(const unsigned* __restrict__ nxt, long steps, unsigned* out) {
  unsigned p = 0;
  for (long s = 0; s < steps; ++s) p = nxt[p];
  out[0] = p;
}

template <int U>
__global__ void stream(const float4* __restrict__ a, long n4, int iters, float* out) {
  const long tid = blockIdx.x * (long)blockDim.x + threadIdx.x, nth = (long)gridDim.x * blockDim.x;
  float s = 0.f;
#pragma unroll 1
  for (int it = 0; it < iters; ++it) {
    float4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) v[u] = a[tid + (long)(it * U + u) * nth];
#pragma unroll
    for (int u = 0; u < U; ++u) s += v[u].x + v[u].y + v[u].z + v[u].w;
  }
  if (s == 1234.5f) out[tid] = s;
}

static float time_ms(void (*f)(void*), void* arg, int reps) {
  gpu(Event_t) a, b; gpu(EventCreate)(&a); gpu(EventCreate)(&b);
  float best = 1e30f;
  for (int r = 0; r < reps; ++r) {
    gpu(EventRecord)(a); f(arg); gpu(EventRecord)(b); gpu(EventSynchronize)(b);
    float ms; gpu(EventElapsedTime)(&ms, a, b); if (r && ms < best) best = ms;
  }
  return best;
}

struct SArg { const float4* a; long n4; int iters, blocks, u; float* out; };
static void run_stream(void* p) {
  SArg* s = (SArg*)p;
  switch (s->u) {
    case 1: stream<1><<<s->blocks, 256>>>(s->a, s->n4, s->iters, s->out); break;
    case 2: stream<2><<<s->blocks, 256>>>(s->a, s->n4, s->iters, s->out); break;
    case 4: stream<4><<<s->blocks, 256>>>(s->a, s->n4, s->iters, s->out); break;
    default: stream<8><<<s->blocks, 256>>>(s->a, s->n4, s->iters, s->out); break;
  }
}
struct CArg { const unsigned* nxt; long steps; unsigned* out; };
static void run_chase(void* p) { CArg* c = (CArg*)p; chase<<<1, 1>>>(c->nxt, c->steps, c->out); }

int main() {
  gpuDeviceProp pr; gpu(GetDeviceProperties)(&pr, 0);
  printf("device %s: SM/CU %d, L2 %d KB, smem/SM %zu KB, smem/block max %zu KB, regs/SM %d, clock %d MHz\n", pr.name,
         pr.multiProcessorCount, pr.l2CacheSize / 1024, (size_t)SMEM_SM(pr) / 1024,
         pr.sharedMemPerBlock / 1024, (int)REGS_SM(pr), pr.clockRate / 1000);
  unsigned* out; gpu(Malloc)(&out, 64 << 20);
  // (2) latency: random single cycle over the working set, 128 B node spacing (one node per cache line)
  for (long ws_mb : {1L, 4L, 16L, 256L, 1024L}) {
    const long nodes = ws_mb * 1024 * 1024 / 128;
    std::vector<unsigned> perm(nodes), nxt(nodes * 32, 0);
    for (long i = 0; i < nodes; ++i) perm[i] = (unsigned)i;
    std::shuffle(perm.begin() + 1, perm.end(), std::mt19937(7));
    for (long i = 0; i < nodes; ++i) nxt[(long)perm[i] * 32] = perm[(i + 1) % nodes] * 32;
    unsigned* d; gpu(Malloc)(&d, nodes * 128); gpu(Memcpy)(d, nxt.data(), nodes * 128, gpuMemcpyHostToDevice);
    CArg c{d, 200000, out};
    const float ms = time_ms(run_chase, &c, 4);
    printf("latency  ws %5ld MB : %7.1f ns per dependent load\n", ws_mb, ms * 1e6 / c.steps);
    gpu(Free)(d);
  }
  // (3) bandwidth vs in-flight bytes, 2 GB buffer
  const long bytes = 2L << 30, n4 = bytes / 16;
  float4* a; gpu(Malloc)(&a, bytes); gpu(Memset)(a, 1, bytes);
  float* o; gpu(Malloc)(&o, 256L << 20);
  const int sm = pr.multiProcessorCount;
  printf("bw: blocks/SM  U  inflight_KB/SM  GB/s\n");
  for (int bps : {1, 2, 4, 8}) {
    for (int u : {1, 2, 4, 8}) {
      SArg s{a, n4, 0, sm * bps, u, o};
      s.iters = (int)(n4 / ((long)s.blocks * 256 * u));
      const float ms = time_ms(run_stream, &s, 5);
      const double moved = (double)s.iters * s.blocks * 256 * u * 16;
      printf("bw: %d %d %7.0f %8.0f\n", bps, u, bps * 256.0 * u * 16 / 1024, moved / (ms * 1e-3) / 1e9);
    }
  }
  return 0;
}
