"""kra P-92 prototype (bitwise): fully asynchronous s4b step pipeline on gfx936 LDS-DMA -- the test of whether a
1-CTA/CU "fat" kernel can hide memory latency the way FLA's kernels do on NV (R61: partial DMA is drained by the
compiler's in-order vmcnt waits; every loop load has to go through DMA with hand-managed waits).

    python3 p92_s4b_async.py <dir holding g2_wy.cuh>

Per step kt the loop used 7 register loads (dwT tile, h / dh tiles kt+2, g row, k row, q row, chunk-end g row) and
3 stores (dq / dk / dg).  Now:
  * every per-step operand of step kt+1 is DMA'd (`buffer_load_dword{x2,x4} ... lds`) at the TOP of step kt into
    LDS slot (kt+1)&1 -- h / dh tiles (16 B/lane, the existing stg slots), the dwT tile (8 B/lane, its P-29 XOR
    swizzle moved to the source addresses), g / chunk-end g rows (16 B/lane), k / q rows (8 B/lane; rows past a
    varlen chunk read out of bounds -> 0, and are masked again after the LDS read);
  * the step reads them back from LDS (lane-linear, the lane's own slot);
  * the dq / dk / dg stores of step kt are issued at the top of step kt+1 (behind the next DMA group), so one
    `s_waitcnt vmcnt(0)` before the end-of-step barrier retires both after a whole step of overlap;
  * no compiler-visible global load remains inside the loop (the prologue's register loads are retired by an
    explicit __builtin_amdgcn_s_waitcnt before it), so the compiler inserts no vmcnt wait in the loop.
LDS: +2 x 14 KB of slots -> ~53 KB, i.e. 1 CTA/CU (the regime under test).  KDA_G2_S4B_PADLDS (P-91 knob) is
included so the register-path tree can be forced to 1 CTA/CU for the comparison.  Values unchanged."""
import sys
from pathlib import Path

f = Path(sys.argv[1]) / "g2_wy.cuh"
s = f.read_text()


def sub(old, new, cnt=1, where=None):
    global s
    tgt = s if where is None else where
    assert tgt.count(old) == cnt, (old[:90], tgt.count(old))
    if where is None:
        s = s.replace(old, new)
    return tgt.replace(old, new)


if "--no-pad" not in sys.argv:   # --no-pad: keep the knob out, so KDA_G2_S4B_PADLDS only throttles the other tree
  sub("""        if (ctok) hipLaunchKernelGGL(p3_wy_s4b_kernel<true>, s4grd, dim3(256), 0, s, pb);
        else hipLaunchKernelGGL(p3_wy_s4b_kernel<false>, s4grd, dim3(256), 0, s, pb);""",
    """        const size_t s4bpad = (size_t)g2::g2_env_long("KDA_G2_S4B_PADLDS", 0);   // kra P-91: occupancy knob
        if (ctok) hipLaunchKernelGGL(p3_wy_s4b_kernel<true>, s4grd, dim3(256), s4bpad, s, pb);
        else hipLaunchKernelGGL(p3_wy_s4b_kernel<false>, s4grd, dim3(256), s4bpad, s, pb);""")
sub("""    bf16 stg[2][2][8 * 256];
    bf16 dwTs[16 * 64];          // kra P-29: dwT tile of step kt, column groups XOR-swizzled by row
};""", """    bf16 stg[2][2][8 * 256];
    bf16 dwTs[16 * 64];          // kra P-29: dwT tile of step kt, column groups XOR-swizzled by row
    struct {                     // kra P-92: the step's DMA'd row operands, lane-linear (thread tid at [tid])
        bf16 dwT[16 * 64];       //   dwT tile, P-29 swizzle applied on the source side
        f4 gv[256], gn[256];     //   g row, chunk-end g row (16 B / thread)
        v4bh kk[256], qq[256];   //   k row, q row (8 B / thread)
    } f[2];
};""")

a = s.index("__device__ __forceinline__ void s4b_tail(")
b = s.index("p3_wy_s4b_kernel(WyS4BParams p) {", a)
pre, body, post = s[:a], s[a:b], s[b:]

body = sub("""    const int4* hg4 = reinterpret_cast<const int4*>(hb);
    const int4* dhg4 = reinterpret_cast<const int4*>(dhb);
    int4* stg4 = reinterpret_cast<int4*>(&sm.stg[0][0][0]);
    #pragma unroll
    for (int t0 = 0; t0 < 2; ++t0) {
        stg4[(t0 * 2 + 0) * 256 + threadIdx.x] = hg4[t0 * 256 + threadIdx.x];
        stg4[(t0 * 2 + 1) * 256 + threadIdx.x] = dhg4[t0 * 256 + threadIdx.x];
    }
    lds_barrier();
""", """    // ---- kra P-92: raw buffer descriptors (gfx9 dword3 0x00020000) and the per-thread DMA source offsets
    typedef int p92_v4i __attribute__((ext_vector_type(4)));
    auto p92_rsrc = [](const void* ptr, unsigned bytes) {
        const unsigned long a_ = (unsigned long)ptr;
        p92_v4i r_;
        r_.x = __builtin_amdgcn_readfirstlane((int)(a_ & 0xffffffffu));
        r_.y = __builtin_amdgcn_readfirstlane((int)(a_ >> 32));
        r_.z = (int)bytes;
        r_.w = 0x00020000;
        return r_;
    };
    const p92_v4i Rh = p92_rsrc(hb, 8 * 4096), Rd = p92_rsrc(dhb, 8 * 4096), Rw = p92_rsrc(dwTg, kD * kBT * 2);
    const p92_v4i Rg = p92_rsrc(g_, 1u << 26), Rk = p92_rsrc(k_, 1u << 26), Rq = p92_rsrc(q_, 1u << 26);
    const int tid_ = threadIdx.x, w_ = tid_ >> 6;
    const int vo16 = tid_ * 16;                                                   // h / dh tile quad
    const int vow = ((tid_ >> 4) * kBT + 4 * ((tid_ & 15) ^ (tid_ >> 4))) * 2;    // dwT: swizzle on the source side
    const int vog = rog * 4, vogn = rgn * 4;                                      // g rows (clamped)
    const int vok = rok ? ro * 2 : 0x7ffffff0;                                    // k / q rows: OOB -> 0
    auto dma = [&](const char* op_unused, int m0, int vo, const p92_v4i& r, int so, int sz) {
        if (sz == 16)
            asm volatile("s_mov_b32 m0, %0\\n\\ts_nop 0\\n\\tbuffer_load_dwordx4 %1, %2, %3 offen lds"
                         :: "s"(m0), "v"(vo), "s"(r), "s"(so) : "memory", "m0");
        else
            asm volatile("s_mov_b32 m0, %0\\n\\ts_nop 0\\n\\tbuffer_load_dwordx2 %1, %2, %3 offen lds"
                         :: "s"(m0), "v"(vo), "s"(r), "s"(so) : "memory", "m0");
    };
    auto lds_at = [](const void* p_, int wofs) { return __builtin_amdgcn_readfirstlane((int)(size_t)p_ + wofs); };
    auto issue = [&](int st, int sl) {   // every operand of step st -> LDS slot sl
        const int z = 0;
        dma("h",  lds_at(&sm.stg[sl][0][0], w_ * 1024), vo16, Rh, __builtin_amdgcn_readfirstlane(st * 4096), 16);
        dma("dh", lds_at(&sm.stg[sl][1][0], w_ * 1024), vo16, Rd, __builtin_amdgcn_readfirstlane(st * 4096), 16);
        dma("w",  lds_at(&sm.f[sl].dwT[0], w_ * 512), vow, Rw, __builtin_amdgcn_readfirstlane(st * 16 * kBT * 2), 8);
        dma("g",  lds_at(&sm.f[sl].gv[0], w_ * 1024), vog + st * 64, Rg, z, 16);
        dma("gn", lds_at(&sm.f[sl].gn[0], w_ * 1024), vogn + st * 64, Rg, z, 16);
        dma("k",  lds_at(&sm.f[sl].kk[0], w_ * 512), rok ? vok + st * 32 : vok, Rk, z, 8);
        dma("q",  lds_at(&sm.f[sl].qq[0], w_ * 512), rok ? vok + st * 32 : vok, Rq, z, 8);
    };
    issue(0, 0);
    __builtin_amdgcn_s_waitcnt(0x0F70);   // vmcnt(0): the prologue's register loads AND step 0's DMA (compiler-visible)
    lds_barrier();
    f4 pq_, pk_, pg_;                      // deferred stores of the previous step
    int pkc_ = 0;
""", where=body)
body = sub("""        const int buf = kt & 1, kc = 16 * kt;
        // kra P-29: this kt's dwT tile, one 8 B load per thread (issued first: oldest in vmcnt)
        const int dwr = threadIdx.x >> 4, dwc = threadIdx.x & 15;
        const v4bh dwv = ld8(dwTg + (kc + dwr) * kBT + 4 * dwc);
        int4 pfh, pfd;
        const bf16* sh = &sm.stg[buf][0][0];
        const bf16* sdh = &sm.stg[buf][1][0];
        auto HB = [&](int vt) { return ld8(sh + vt * 256 + l * 4); };
        auto DHB = [&](int vt) { return ld8(sdh + vt * 256 + l * 4); };
        if (kt + 2 < 8) {
            pfh = hg4[(kt + 2) * 256 + threadIdx.x];
            pfd = dhg4[(kt + 2) * 256 + threadIdx.x];
        }
        const f4 gv = ld16f(g_ + rog + kc);
        const v4bh kk = rok ? ld8(k_ + ro + kc) : v4bh{0, 0, 0, 0};
        const v4bh qq = rok ? ld8(q_ + ro + kc) : v4bh{0, 0, 0, 0};
        const f4 gn = ld16f(g_ + rgn + kc);
""", """        const int buf = kt & 1, kc = 16 * kt;
        if (kt + 1 < 8) issue(kt + 1, (kt + 1) & 1);   // kra P-92: step kt+1 in flight for the whole of step kt
        if (kt > 0 && rok) {                             // kra P-92: step kt-1's stores, behind the DMA group
            *reinterpret_cast<f4*>(dq_ + ro + pkc_) = pq_;
            *reinterpret_cast<f4*>(dk_ + ro + pkc_) = pk_;
            *reinterpret_cast<f4*>(dg_ + ro + pkc_) = pg_;
        }
        const bf16* sh = &sm.stg[buf][0][0];
        const bf16* sdh = &sm.stg[buf][1][0];
        auto HB = [&](int vt) { return ld8(sh + vt * 256 + l * 4); };
        auto DHB = [&](int vt) { return ld8(sdh + vt * 256 + l * 4); };
        const f4 gv = sm.f[buf].gv[tid_];
        const v4bh kk = rok ? sm.f[buf].kk[tid_] : v4bh{0, 0, 0, 0};
        const v4bh qq = rok ? sm.f[buf].qq[tid_] : v4bh{0, 0, 0, 0};
        const f4 gn = sm.f[buf].gn[tid_];
""", where=body)
body = sub("""        *reinterpret_cast<v4bh*>(&sm.dwTs[dwr * 64 + 4 * (dwc ^ dwr)]) = dwv;   // kra P-29
        lds_barrier();
        if (kt + 2 < 8) {                          // tile kt+2 -> the buffer tile kt freed
            stg4[(buf * 2 + 0) * 256 + threadIdx.x] = pfh;
            stg4[(buf * 2 + 1) * 256 + threadIdx.x] = pfd;
        }""", """        lds_barrier();                             // kra P-92: dwT arrived by DMA (slot buf), no stg refill here""", where=body)
body = sub("""            acc = mmac(aa[st], ld8(&sm.dwTs[pm * 64 + 4 * ((4 * st + q) ^ pm)]), acc);   // kra P-29""",
           """            acc = mmac(aa[st], ld8(&sm.f[buf].dwT[pm * 64 + 4 * ((4 * st + q) ^ pm)]), acc);   // kra P-29 / P-92""",
           where=body)
body = sub("""        if (rok) {
            *reinterpret_cast<f4*>(dq_ + ro + kc) = oq;
            *reinterpret_cast<f4*>(dk_ + ro + kc) = ok;
            *reinterpret_cast<f4*>(dg_ + ro + kc) = og;
        }
        lds_barrier();                             // table WAR: kt's reads done before kt+1 writes
    }""", """        pq_ = oq; pk_ = ok; pg_ = og; pkc_ = kc;    // kra P-92: stored at the top of step kt+1
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");   // kra P-92: step kt+1's DMA group + step kt-1's stores
        lds_barrier();                             // table WAR: kt's reads done before kt+1 writes
    }
    if (rok) {                                     // kra P-92: the last step's stores
        *reinterpret_cast<f4*>(dq_ + ro + pkc_) = pq_;
        *reinterpret_cast<f4*>(dk_ + ro + pkc_) = pk_;
        *reinterpret_cast<f4*>(dg_ + ro + pkc_) = pg_;
    }""", where=body)
s = pre + body + post
f.write_text(s)
print("P-92 APPLIED")
