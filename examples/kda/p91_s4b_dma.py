"""kra P-91 prototype (bitwise): s4b stages its per-kt h / dh tiles with gfx936's direct-to-LDS buffer loads
(`buffer_load_dwordx4 ... lds`, verified in blocks/lds_dma*.hip) instead of global -> VGPR -> ds_write.

    python3 p91_s4b_dma.py <dir holding g2_wy.cuh> [--pad-only]

--tri      : DMA staging with a third LDS slot (+8 KB, 1 CTA/CU only): tile kt+2 is issued at the top of step kt,
             right after the step's own loads, into slot (kt+2)%3 (freed at step kt-1's table barrier), so it has ~2
             steps to land; vmcnt(2) before the table barrier keeps that DMA in flight.
--pad-only : only add the occupancy knob KDA_G2_S4B_PADLDS=<bytes> (dynamic LDS per CTA, P-78a method) so the
             2 -> 1 CTA/CU regime can be compared on the same source.
default    : the knob + the DMA staging.  Tile kt+2 (h and dh, 16 B per lane each) is issued into the buffer tile kt
             freed, right after the table barrier (where the register path stored it); M0 = the wave's 1 KB slice of
             the buffer, lane l lands at M0 + 16 l == stg4[... * 256 + tid] of the register path.  Completion: an
             explicit vmcnt(0) before the next table barrier (by then every other load of that iteration has been
             consumed), so tile kt+2 is visible to all waves from iteration kt+1's table barrier on.  The prologue
             (tiles 0, 1) keeps the register path.  Same bytes, same LDS image -> bit-identical."""
import sys
from pathlib import Path

f = Path(sys.argv[1]) / "g2_wy.cuh"
pad_only = "--pad-only" in sys.argv
tri = "--tri" in sys.argv   # 3 LDS slots: tile kt+2 issued at the top of kt (after the step's own loads), ~2 steps ahead
s = f.read_text()


def sub(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (old[:90], s.count(old))
    s = s.replace(old, new)


sub("""        if (ctok) hipLaunchKernelGGL(p3_wy_s4b_kernel<true>, s4grd, dim3(256), 0, s, pb);
        else hipLaunchKernelGGL(p3_wy_s4b_kernel<false>, s4grd, dim3(256), 0, s, pb);""",
    """        const size_t s4bpad = (size_t)g2::g2_env_long("KDA_G2_S4B_PADLDS", 0);   // kra P-91: occupancy knob
        if (ctok) hipLaunchKernelGGL(p3_wy_s4b_kernel<true>, s4grd, dim3(256), s4bpad, s, pb);
        else hipLaunchKernelGGL(p3_wy_s4b_kernel<false>, s4grd, dim3(256), s4bpad, s, pb);""")
if tri:
    sub("""    bf16 stg[2][2][8 * 256];
    bf16 dwTs[16 * 64];          // kra P-29""", """    bf16 stg[3][2][8 * 256];   // kra P-91 tri: third slot
    bf16 dwTs[16 * 64];          // kra P-29""")
if not pad_only:
    # resource descriptors for the h / dh blocks of this (chunk, head), once per CTA
    sub("""    const int4* hg4 = reinterpret_cast<const int4*>(hb);
    const int4* dhg4 = reinterpret_cast<const int4*>(dhb);
    int4* stg4 = reinterpret_cast<int4*>(&sm.stg[0][0][0]);
    #pragma unroll
    for (int t0 = 0; t0 < 2; ++t0) {
        stg4[(t0 * 2 + 0) * 256 + threadIdx.x] = hg4[t0 * 256 + threadIdx.x];
        stg4[(t0 * 2 + 1) * 256 + threadIdx.x] = dhg4[t0 * 256 + threadIdx.x];
    }
    lds_barrier();
""", """    const int4* hg4 = reinterpret_cast<const int4*>(hb);
    const int4* dhg4 = reinterpret_cast<const int4*>(dhb);
    int4* stg4 = reinterpret_cast<int4*>(&sm.stg[0][0][0]);
    #pragma unroll
    for (int t0 = 0; t0 < 2; ++t0) {
        stg4[(t0 * 2 + 0) * 256 + threadIdx.x] = hg4[t0 * 256 + threadIdx.x];
        stg4[(t0 * 2 + 1) * 256 + threadIdx.x] = dhg4[t0 * 256 + threadIdx.x];
    }
    lds_barrier();
    // kra P-91: raw buffer descriptors (gfx9 dword3 0x00020000) over the 8 tiles x 4 KB of h and dh
    typedef int p91_v4i __attribute__((ext_vector_type(4)));
    auto p91_rsrc = [](const void* ptr) {
        const unsigned long a_ = (unsigned long)ptr;
        p91_v4i r_;
        r_.x = __builtin_amdgcn_readfirstlane((int)(a_ & 0xffffffffu));
        r_.y = __builtin_amdgcn_readfirstlane((int)(a_ >> 32));
        r_.z = 8 * 4096;
        r_.w = 0x00020000;
        return r_;
    };
    const p91_v4i rsh = p91_rsrc(hb), rsd = p91_rsrc(dhb);
    const int p91_voff = (int)threadIdx.x * 16;
    const int p91_wofs = (int)(threadIdx.x >> 6) * 1024;
""")
    sub("""        int4 pfh, pfd;
        const bf16* sh = &sm.stg[buf][0][0];
        const bf16* sdh = &sm.stg[buf][1][0];
        auto HB = [&](int vt) { return ld8(sh + vt * 256 + l * 4); };
        auto DHB = [&](int vt) { return ld8(sdh + vt * 256 + l * 4); };
        if (kt + 2 < 8) {
            pfh = hg4[(kt + 2) * 256 + threadIdx.x];
            pfd = dhg4[(kt + 2) * 256 + threadIdx.x];
        }
""", """        const bf16* sh = &sm.stg[buf][0][0];
        const bf16* sdh = &sm.stg[buf][1][0];
        auto HB = [&](int vt) { return ld8(sh + vt * 256 + l * 4); };
        auto DHB = [&](int vt) { return ld8(sdh + vt * 256 + l * 4); };
""")
    sub("""        *reinterpret_cast<v4bh*>(&sm.dwTs[dwr * 64 + 4 * (dwc ^ dwr)]) = dwv;   // kra P-29
        lds_barrier();
        if (kt + 2 < 8) {                          // tile kt+2 -> the buffer tile kt freed
            stg4[(buf * 2 + 0) * 256 + threadIdx.x] = pfh;
            stg4[(buf * 2 + 1) * 256 + threadIdx.x] = pfd;
        }""", """        *reinterpret_cast<v4bh*>(&sm.dwTs[dwr * 64 + 4 * (dwc ^ dwr)]) = dwv;   // kra P-29
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");   // kra P-91: tile kt+1's DMA (issued in kt-1) has landed
        lds_barrier();
        if (kt + 2 < 8) {   // kra P-91: tile kt+2 straight into the buffer tile kt freed (no VGPR, no ds_write)
            const int m0h = __builtin_amdgcn_readfirstlane((int)(size_t)&sm.stg[buf][0][0] + p91_wofs);
            const int m0d = __builtin_amdgcn_readfirstlane((int)(size_t)&sm.stg[buf][1][0] + p91_wofs);
            const int soff = __builtin_amdgcn_readfirstlane((kt + 2) * 4096);
            asm volatile("s_mov_b32 m0, %0\\n\\ts_nop 0\\n\\tbuffer_load_dwordx4 %1, %2, %3 offen lds"
                         :: "s"(m0h), "v"(p91_voff), "s"(rsh), "s"(soff) : "memory", "m0");
            asm volatile("s_mov_b32 m0, %0\\n\\ts_nop 0\\n\\tbuffer_load_dwordx4 %1, %2, %3 offen lds"
                         :: "s"(m0d), "v"(p91_voff), "s"(rsd), "s"(soff) : "memory", "m0");
        }""")
if tri:   # all tri edits are confined to s4b_tail (the fused s4 kernel has look-alike lines)
    _a = s.index("__device__ __forceinline__ void s4b_tail(")
    _b = s.index("p3_wy_s4b_kernel(WyS4BParams p) {", _a)
    _pre, s, _post = s[:_a], s[_a:_b], s[_b:]
    sub("""        const bf16* sh = &sm.stg[buf][0][0];
        const bf16* sdh = &sm.stg[buf][1][0];""", """        const int sbuf = kt % 3;   // kra P-91 tri: stg slot (cs1/cs2 keep the kt&1 double buffer)
        const bf16* sh = &sm.stg[sbuf][0][0];
        const bf16* sdh = &sm.stg[sbuf][1][0];""")
    sub("""        const f4 gn = ld16f(g_ + rgn + kc);
        f4 dkp, dgK;""", """        const f4 gn = ld16f(g_ + rgn + kc);
        if (kt + 2 < 8) {   // kra P-91 tri: tile kt+2 -> slot (kt+2)%3, after this step's own loads
            const int nb = (kt + 2) % 3;
            const int m0h = __builtin_amdgcn_readfirstlane((int)(size_t)&sm.stg[nb][0][0] + p91_wofs);
            const int m0d = __builtin_amdgcn_readfirstlane((int)(size_t)&sm.stg[nb][1][0] + p91_wofs);
            const int soff = __builtin_amdgcn_readfirstlane((kt + 2) * 4096);
            asm volatile("s_mov_b32 m0, %0\\n\\ts_nop 0\\n\\tbuffer_load_dwordx4 %1, %2, %3 offen lds"
                         :: "s"(m0h), "v"(p91_voff), "s"(rsh), "s"(soff) : "memory", "m0");
            asm volatile("s_mov_b32 m0, %0\\n\\ts_nop 0\\n\\tbuffer_load_dwordx4 %1, %2, %3 offen lds"
                         :: "s"(m0d), "v"(p91_voff), "s"(rsd), "s"(soff) : "memory", "m0");
        }
        f4 dkp, dgK;""")
    sub("""        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");   // kra P-91: tile kt+1's DMA (issued in kt-1) has landed""",
        """        if (kt + 2 < 8) asm volatile("s_waitcnt vmcnt(2)" ::: "memory");   // kra P-91 tri: tile kt+1 landed, kt+2 in flight
        else            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");""")
    a = s.index("        if (kt + 2 < 8) {   // kra P-91: tile kt+2 straight into the buffer tile kt freed")
    b = s.index("        }", s.index(":: \"s\"(m0d)", a)) + len("        }")
    s = s[:a] + "        // kra P-91 tri: tile kt+2 was issued at the top of the step" + s[b:]
    s = _pre + s + _post
f.write_text(s)
print("P-91 APPLIED" + (" (pad only)" if pad_only else "") + (" (tri)" if tri else ""))
