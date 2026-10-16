"""kra P-83 probe (bitwise): s4a issues its front-phase and pass-W global loads up front (template int PF,
KDA_G2_S4A_PF=1).

SQTT on P-82b (results/kda-ceiling-cards-20261016-p82 §3): s4a waits on VMEM 37-38% of its wave-cycles.  The largest
site is pass W's per-kt vmcnt(1) (7.4% at 16K/12, 12.7% at 8K/48): the h block / kgB tile of kt+1 is loaded during
kt-1 and stored to LDS at the top of kt, one short step (12 MMAC) later.  Every phase entry adds one more 3-4% wait
(A staging, dva, the dv2 loop's v rows, the h/kg prologue) because each load is consumed right after it is issued.
PF=1 issues, right after the two beta loads and in consumption order (vmcnt retires in order), the A block quads,
the dvN fragments, the v row quads, all 8 h blocks (16 B/thread each) and all 8 kgB tiles (8 B/thread each); the
phases then consume registers, pass W is fully unrolled (static indices, R55) and only stores its next block/tile to
the same double-buffered LDS slots.  Values and MMAC order are unchanged.
    python3 p83_s4a_pf.py <dir holding g2_wy.cuh>"""
import sys
from pathlib import Path

f = Path(sys.argv[1]) / "g2_wy.cuh"
s = f.read_text()


def sub(b, old, new, cnt=1):
    assert b.count(old) == cnt, (old[:90], b.count(old))
    return b.replace(old, new)


s = sub(s, """template <bool VL, bool AFS = false>   // kra P-28c: AFS = stage the dA-stage A block in XT
__global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(1, 3)))
p3_wy_s4a_kernel(WyS4AParams p_) {""", """template <bool VL, bool AFS = false, int PF = 0>   // kra P-28c: AFS = stage the dA-stage A block in XT; P-83: PF
__global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(1, 3)))
p3_wy_s4a_kernel(WyS4AParams p_) {""")
head, rest = s.split("p3_wy_s4a_kernel(WyS4AParams p_) {", 1)
body, tail = rest.split("template <int A, int M", 1)
body = sub(body, """            v4bh av_[4];                                   // kra P-43: all four loads in flight
            #pragma unroll
            for (int it_ = 0; it_ < 4; ++it_) {
                const int j = tid + 256 * it_, row = j >> 4, qc = j & 15;
                // VL: rows past the chunk's end are NOT in the packed buffer -- never load them
                // (their aa entries are masked below anyway).
                av_[it_] = g2_row_ok<M>(row, cl)
                    ? *reinterpret_cast<const v4bh*>(A_ + row * RA + 4 * qc) : v4bh{0, 0, 0, 0};
            }""", """            v4bh av_[4];                                   // kra P-43: all four loads in flight
            if constexpr (PF) {                            // kra P-83: issued up front
                #pragma unroll
                for (int it_ = 0; it_ < 4; ++it_) av_[it_] = pav[it_];
            } else {
            #pragma unroll
            for (int it_ = 0; it_ < 4; ++it_) {
                const int j = tid + 256 * it_, row = j >> 4, qc = j & 15;
                // VL: rows past the chunk's end are NOT in the packed buffer -- never load them
                // (their aa entries are masked below anyway).
                av_[it_] = g2_row_ok<M>(row, cl)
                    ? *reinterpret_cast<const v4bh*>(A_ + row * RA + 4 * qc) : v4bh{0, 0, 0, 0};
            }
            }""")
body = sub(body, """        const float beta_t = trok ? beta_[tr * H] : 0.f;
        if (tid < kBT) BETAs[tid] = g2_row_ok<M>(tid, cl) ? beta_[tid * H] : 0.f;   // kra P-28""",
           """        const float beta_t = trok ? beta_[tr * H] : 0.f;
        // kra P-83 PF: the beta loads first, then every front-phase / pass-W operand in consumption order
        v4bh pav[4], pdva[8], pvrow[8], pkg[8];
        f4 phb[8];
        if constexpr (PF) {
            const float bts_ = (tid < kBT && g2_row_ok<M>(tid, cl)) ? beta_[tid * H] : 0.f;
            #pragma unroll
            for (int it_ = 0; it_ < 4; ++it_) {
                const int j = tid + 256 * it_, row = j >> 4, qc = j & 15;
                pav[it_] = g2_row_ok<M>(row, cl)
                    ? *reinterpret_cast<const v4bh*>(A_ + row * RA + 4 * qc) : v4bh{0, 0, 0, 0};
            }
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) pdva[vt] = aN(dvN, vt);
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) pvrow[vt] = trok ? ld8(v_ + ro + 16 * vt) : v4bh{0, 0, 0, 0};
            const bf16* hsrc0 = hb + 8 * tid;
            const bf16* kgsrc0 = kgB + (((tid >> 6) * 8) * 64 + (tid & 63)) * 4;
            #pragma unroll
            for (int kt = 0; kt < 8; ++kt) {
                phb[kt] = *reinterpret_cast<const f4*>(hsrc0 + kt * 2048);
                pkg[kt] = ld8(kgsrc0 + kt * 256);
            }
            if (tid < kBT) BETAs[tid] = bts_;
        } else {
        if (tid < kBT) BETAs[tid] = g2_row_ok<M>(tid, cl) ? beta_[tid * H] : 0.f;   // kra P-28
        }""")
body = sub(body, """        for (int vt = 0; vt < 8; ++vt) dva[vt] = aN(dvN, vt);""",
           """        for (int vt = 0; vt < 8; ++vt) dva[vt] = PF ? pdva[vt] : aN(dvN, vt);   // kra P-83""")
body = sub(body, """            const v4bh vv = trok ? ld8(v_ + ro + 16 * vt) : v4bh{0, 0, 0, 0};""",
           """            const v4bh vv = PF ? pvrow[vt] : (trok ? ld8(v_ + ro + 16 * vt) : v4bh{0, 0, 0, 0});   // kra P-83""")
body = sub(body, """            // kra P-28: thread tid carries fragment (st = tid>>6, lane = tid&63) of the kgB tile""",
           """            if constexpr (PF) {   // kra P-83: blocks / tiles already in registers; pass W fully unrolled
                short* hs = reinterpret_cast<short*>(XT);
                *reinterpret_cast<v4bh*>(&KGs[0][0] + 4 * tid) = pkg[0];
                *reinterpret_cast<f4*>(hs + 8 * tid) = phb[0];
                lds_barrier();                                                  // block 0 visible
                #pragma unroll
                for (int kt = 0; kt < 8; ++kt) {
                    v4bh bf[8];
                    const bf16* hk = reinterpret_cast<const bf16*>(hs + (kt & 1) * 2048) + l * 4;
                    #pragma unroll
                    for (int vt = 0; vt < 8; ++vt) bf[vt] = ld8(hk + vt * 256);
                    if (kt < 7) {
                        *reinterpret_cast<f4*>(hs + ((kt + 1) & 1) * 2048 + 8 * tid) = phb[kt + 1];
                        *reinterpret_cast<v4bh*>(&KGs[(kt + 1) & 1][0] + 4 * tid) = pkg[kt + 1];
                    }
                    body(kt, bf);
                    lds_barrier();      // block kt+1 visible; reads of block kt done before reuse
                }
            } else {
            // kra P-28: thread tid carries fragment (st = tid>>6, lane = tid&63) of the kgB tile""")
body = sub(body, """                body(kt, bf);
                lds_barrier();      // block kt+1 visible; reads of block kt done before reuse
            }
""", """                body(kt, bf);
                lds_barrier();      // block kt+1 visible; reads of block kt done before reuse
            }
            }   // kra P-83: !PF
""")
s = head + "p3_wy_s4a_kernel(WyS4AParams p_) {" + body + "template <int A, int M" + tail
s = sub(s, """        if (ctok) {
            if (afs) hipLaunchKernelGGL((p3_wy_s4a_kernel<true, true>), dim3(NT, H), dim3(256), 0, s, pa);
            else     hipLaunchKernelGGL((p3_wy_s4a_kernel<true, false>), dim3(NT, H), dim3(256), 0, s, pa);
        } else {
            if (afs) hipLaunchKernelGGL((p3_wy_s4a_kernel<false, true>), dim3(NT, H), dim3(256), 0, s, pa);
            else     hipLaunchKernelGGL((p3_wy_s4a_kernel<false, false>), dim3(NT, H), dim3(256), 0, s, pa);
        }""", """        const bool pf = g2::g2_env_long("KDA_G2_S4A_PF", 0) != 0;   // kra P-83 probe
        if (ctok) {
            if (afs) { if (pf) hipLaunchKernelGGL((p3_wy_s4a_kernel<true, true, 1>), dim3(NT, H), dim3(256), 0, s, pa);
                       else    hipLaunchKernelGGL((p3_wy_s4a_kernel<true, true>), dim3(NT, H), dim3(256), 0, s, pa); }
            else     { if (pf) hipLaunchKernelGGL((p3_wy_s4a_kernel<true, false, 1>), dim3(NT, H), dim3(256), 0, s, pa);
                       else    hipLaunchKernelGGL((p3_wy_s4a_kernel<true, false>), dim3(NT, H), dim3(256), 0, s, pa); }
        } else {
            if (afs) { if (pf) hipLaunchKernelGGL((p3_wy_s4a_kernel<false, true, 1>), dim3(NT, H), dim3(256), 0, s, pa);
                       else    hipLaunchKernelGGL((p3_wy_s4a_kernel<false, true>), dim3(NT, H), dim3(256), 0, s, pa); }
            else     { if (pf) hipLaunchKernelGGL((p3_wy_s4a_kernel<false, false, 1>), dim3(NT, H), dim3(256), 0, s, pa);
                       else    hipLaunchKernelGGL((p3_wy_s4a_kernel<false, false>), dim3(NT, H), dim3(256), 0, s, pa); }
        }""")
f.write_text(s)
print("P-83 APPLIED")
