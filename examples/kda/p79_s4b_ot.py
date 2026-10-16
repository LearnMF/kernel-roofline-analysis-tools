"""kra P-79 probe (bitwise): an occupancy-tolerant s4b kt loop -- step 1 of the s4b+tail segment fusion.

results/kda-segfusion-20261016: fusing tail into s4b needs ~96 more VGPRs -> 1 CTA/CU, which costs s4b +355 us at
16K/12 (P-78a).  SQTT of s4b at 1 CTA/CU: per wave 61.8K cycles = issue 25.1K + waitcnt 19.0K + issue-delay 13.3K +
barrier 4.4K; the largest exposed wait is one vmcnt(3) per kt (~12% of wave-cycles): the first use of gv / gn (pass K's
decay) waits behind the in-order dwv, pfh, pfd loads issued before them.  Template int OT on s4b (KDA_G2_S4B_OT):
  1  per-kt loads reordered: gv, gn, kk, qq first, then dwv, then the kt+2 h/dh stage (pfh, pfd)
  2  OT 1's order + gv / gn / kk / qq / dwv of kt+1 issued in kt right after their last use (same registers:
     no loop-carried copy, R56), so pass K never waits on this kt's loads
Combine with P-78a's KDA_G2_S4B_PADLDS to measure at 1 CTA/CU.     python3 p79_s4b_ot.py <tree>  (after p78a)"""
import sys
from pathlib import Path

f = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wy.cuh"
s = f.read_text()


def sub(b, old, new, cnt=1):
    assert b.count(old) == cnt, (old[:90], b.count(old))
    return b.replace(old, new)


s = sub(s, """template <int A, int M>
__device__ __forceinline__ void s4b_tail(""", """template <int A, int M, int OT = 0>   // kra P-79: OT
__device__ __forceinline__ void s4b_tail(""")
head, rest = s.split("__device__ __forceinline__ void s4b_tail(", 1)
body, tail = rest.split("p3_wy_s4b_kernel(WyS4BParams p) {", 1)
body = sub(body, """    float dbacc = 0.f;
    #pragma unroll 1
    for (int kt = 0; kt < 8; ++kt) {""", """    float dbacc = 0.f;
    // kra P-79 OT=2: kt's per-tile operands, issued during kt-1 right after their last use
    v4bh dwvN = v4bh{0, 0, 0, 0}, kkN = dwvN, qqN = dwvN;
    f4 gvN = f4{0.f, 0.f, 0.f, 0.f}, gnN = gvN;
    if constexpr (OT == 2) {
        gvN = ld16f(g_ + rog);
        gnN = ld16f(g_ + rgn);
        kkN = rok ? ld8(k_ + ro) : v4bh{0, 0, 0, 0};
        qqN = rok ? ld8(q_ + ro) : v4bh{0, 0, 0, 0};
        dwvN = ld8(dwTg + (threadIdx.x >> 4) * kBT + 4 * (threadIdx.x & 15));
    }
    #pragma unroll 1
    for (int kt = 0; kt < 8; ++kt) {""")
body = sub(body, """        const v4bh dwv = ld8(dwTg + (kc + dwr) * kBT + 4 * dwc);
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
        const f4 gn = ld16f(g_ + rgn + kc);""", """        v4bh dwv, kk, qq;
        f4 gv, gn;
        int4 pfh, pfd;
        const bf16* sh = &sm.stg[buf][0][0];
        const bf16* sdh = &sm.stg[buf][1][0];
        auto HB = [&](int vt) { return ld8(sh + vt * 256 + l * 4); };
        auto DHB = [&](int vt) { return ld8(sdh + vt * 256 + l * 4); };
        if constexpr (OT == 0) {   // production order
            dwv = ld8(dwTg + (kc + dwr) * kBT + 4 * dwc);
            if (kt + 2 < 8) {
                pfh = hg4[(kt + 2) * 256 + threadIdx.x];
                pfd = dhg4[(kt + 2) * 256 + threadIdx.x];
            }
            gv = ld16f(g_ + rog + kc);
            kk = rok ? ld8(k_ + ro + kc) : v4bh{0, 0, 0, 0};
            qq = rok ? ld8(q_ + ro + kc) : v4bh{0, 0, 0, 0};
            gn = ld16f(g_ + rgn + kc);
        } else if constexpr (OT == 1) {   // kra P-79: first-needed loads first
            gv = ld16f(g_ + rog + kc);
            gn = ld16f(g_ + rgn + kc);
            kk = rok ? ld8(k_ + ro + kc) : v4bh{0, 0, 0, 0};
            qq = rok ? ld8(q_ + ro + kc) : v4bh{0, 0, 0, 0};
            dwv = ld8(dwTg + (kc + dwr) * kBT + 4 * dwc);
            if (kt + 2 < 8) {
                pfh = hg4[(kt + 2) * 256 + threadIdx.x];
                pfd = dhg4[(kt + 2) * 256 + threadIdx.x];
            }
        } else {                          // kra P-79 OT=2: issued during kt-1
            gv = gvN; gn = gnN; kk = kkN; qq = qqN; dwv = dwvN;
            if (kt + 2 < 8) {
                pfh = hg4[(kt + 2) * 256 + threadIdx.x];
                pfd = dhg4[(kt + 2) * 256 + threadIdx.x];
            }
        }""")
body = sub(body, """        *reinterpret_cast<v4bh*>(&sm.dwTs[dwr * 64 + 4 * (dwc ^ dwr)]) = dwv;   // kra P-29
        lds_barrier();""", """        *reinterpret_cast<v4bh*>(&sm.dwTs[dwr * 64 + 4 * (dwc ^ dwr)]) = dwv;   // kra P-29
        if constexpr (OT == 2)
            if (kt + 1 < 8) dwvN = ld8(dwTg + (kc + 16 + dwr) * kBT + 4 * dwc);   // kra P-79: kt+1
        lds_barrier();""")
body = sub(body, """        if (rok) {
            *reinterpret_cast<f4*>(dq_ + ro + kc) = oq;
            *reinterpret_cast<f4*>(dk_ + ro + kc) = ok;
            *reinterpret_cast<f4*>(dg_ + ro + kc) = og;
        }
        lds_barrier();                             // table WAR: kt's reads done before kt+1 writes
    }""", """        if (rok) {
            *reinterpret_cast<f4*>(dq_ + ro + kc) = oq;
            *reinterpret_cast<f4*>(dk_ + ro + kc) = ok;
            *reinterpret_cast<f4*>(dg_ + ro + kc) = og;
        }
        if constexpr (OT == 2)
            if (kt + 1 < 8) {   // kra P-79: kt+1's tile operands, after their last use in kt
                gvN = ld16f(g_ + rog + kc + 16);
                gnN = ld16f(g_ + rgn + kc + 16);
                kkN = rok ? ld8(k_ + ro + kc + 16) : v4bh{0, 0, 0, 0};
                qqN = rok ? ld8(q_ + ro + kc + 16) : v4bh{0, 0, 0, 0};
            }
        lds_barrier();                             // table WAR: kt's reads done before kt+1 writes
    }""")
s = head + "__device__ __forceinline__ void s4b_tail(" + body + "p3_wy_s4b_kernel(WyS4BParams p) {" + tail
s = sub(s, """template <bool VL>
__global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(1, 3)))
p3_wy_s4b_kernel(WyS4BParams p) {""", """template <bool VL, int OT = 0>   // kra P-79
__global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(1, 3)))
p3_wy_s4b_kernel(WyS4BParams p) {""")
for a_ in ("0", "1", "2"):
    s = sub(s, f"case {a_}: s4b_tail<{a_}, M>(p, sm, aa, db0, beta_t, tok0, cl); break;",
            f"case {a_}: s4b_tail<{a_}, M, OT>(p, sm, aa, db0, beta_t, tok0, cl); break;")
s = sub(s, "default: s4b_tail<3, M>(p, sm, aa, db0, beta_t, tok0, cl); break;",
        "default: s4b_tail<3, M, OT>(p, sm, aa, db0, beta_t, tok0, cl); break;")
s = sub(s, """        if (ctok) hipLaunchKernelGGL(p3_wy_s4b_kernel<true>, dim3(NT, H), dim3(256), padl, s, pb);
        else hipLaunchKernelGGL(p3_wy_s4b_kernel<false>, dim3(NT, H), dim3(256), padl, s, pb);""",
        """        const long ot = g2::g2_env_long("KDA_G2_S4B_OT", 0);   // kra P-79 probe
        if (ctok) hipLaunchKernelGGL(p3_wy_s4b_kernel<true>, dim3(NT, H), dim3(256), padl, s, pb);
        else if (ot == 1) hipLaunchKernelGGL((p3_wy_s4b_kernel<false, 1>), dim3(NT, H), dim3(256), padl, s, pb);
        else if (ot == 2) hipLaunchKernelGGL((p3_wy_s4b_kernel<false, 2>), dim3(NT, H), dim3(256), padl, s, pb);
        else hipLaunchKernelGGL(p3_wy_s4b_kernel<false>, dim3(NT, H), dim3(256), padl, s, pb);""")
f.write_text(s)
print("P-79 APPLIED")
