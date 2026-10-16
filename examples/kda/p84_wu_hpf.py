"""kra P-84 probe (bitwise): wu VNEW issues its bN(h) loads ahead of use (template bool HPF, KDA_G2_WU_HPF=1).

SQTT on P-82b (results/kda-ceiling-cards-20261016-p82 §3): wu waits on VMEM 55% of its wave-cycles; 15.2% is one
vmcnt(1) per hB fill load -- P-82's VNEW loads each bN quad and scatters it to LDS right away, after the u-phase MMAC.
HPF: the first V-slab pair's 8 quads per thread are issued right after the phase-2 vbT fill (they land during the
u-phase MMAC chain), the second pair's right after the first pair's fill (they land during the first pair's GEMM).
+16 VGPR per in-flight set (static indices: the pr loop is unrolled, R55).  Values and MMAC order unchanged.
    python3 p84_wu_hpf.py <dir holding g2_wu.cuh>"""
import sys
from pathlib import Path

f = Path(sys.argv[1]) / "g2_wu.cuh"
s = f.read_text()


def sub(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (old[:90], s.count(old))
    s = s.replace(old, new)


sub("""template <bool VL>
__global__ void __launch_bounds__(256) g2_wu_kernel(WuParams p_) {""",
    """template <bool VL, bool HPF = false>   // kra P-84: HPF
__global__ void __launch_bounds__(256) g2_wu_kernel(WuParams p_) {""")
sub("""        v4bh wA[8];   // kra P-82: the w phase's A fragments (== fwd_h_bn's wf for this wave / lane)""",
    """        v4bh wA[8];   // kra P-82: the w phase's A fragments (== fwd_h_bn's wf for this wave / lane)
        v4bh hpre[2][8];   // kra P-84: bN quads of V-slab pair 0 / 1, issued ahead of the fill
        auto hbn_ld = [&](int pr, int it) {   // the VNEW fill's quad (pr, it) of this thread
            const int jj = tid + 256 * it;
            const int ln = jj & 63, vt = (jj >> 6) & 1, kb = (jj >> 7) & 7, sp = jj >> 10;
            return *reinterpret_cast<const v4bh*>(
                p.HbN + ((((size_t)i * p.H + hh) * 8 + kb) * 8 + (2 * (2 * pr + sp) + vt)) * 256 + ln * 4);
        };""")
sub("""                        const v4bh val = *reinterpret_cast<const v4bh*>(
                            p.HbN + ((((size_t)i * p.H + hh) * 8 + kb) * 8 + (2 * vs + vt)) * 256 + ln * 4);""",
    """                        const v4bh val = HPF ? hpre[pr][it] : *reinterpret_cast<const v4bh*>(
                            p.HbN + ((((size_t)i * p.H + hh) * 8 + kb) * 8 + (2 * vs + vt)) * 256 + ln * 4);""")
sub("""                            hb[sp * 32 * kPadK + (16 * vt + 4 * qq + e) * kPadK + 16 * kb + pi16(mm)] = val[e];
                    }
                    __syncthreads();""", """                            hb[sp * 32 * kPadK + (16 * vt + 4 * qq + e) * kPadK + 16 * kb + pi16(mm)] = val[e];
                    }
                    if constexpr (HPF)   // kra P-84: pair 1's quads land during pair 0's GEMM
                        if (pr == 0) {
                            #pragma unroll
                            for (int it = 0; it < 8; ++it) hpre[1][it] = hbn_ld(1, it);
                        }
                    __syncthreads();""")
sub("""        if (p.stop == 4) return;""", """        if constexpr (HPF)   // kra P-84: pair 0's quads land during the u-phase MMAC chain
            if (p.VnOut != nullptr) {
                #pragma unroll
                for (int it = 0; it < 8; ++it) hpre[0][it] = hbn_ld(0, it);
            }
        if (p.stop == 4) return;""")
sub("""    if (ctok)
        hipLaunchKernelGGL(g2::g2_wu_kernel<true>, dim3(T / 64, H), dim3(256), 0, s, p);
    else
        hipLaunchKernelGGL(g2::g2_wu_kernel<false>, dim3(T / 64, H), dim3(256), 0, s, p);""",
    """    const bool hpf = g2::g2_env_long("KDA_G2_WU_HPF", 0) != 0;   // kra P-84 probe
    if (ctok) {
        if (hpf) hipLaunchKernelGGL((g2::g2_wu_kernel<true, true>), dim3(T / 64, H), dim3(256), 0, s, p);
        else     hipLaunchKernelGGL(g2::g2_wu_kernel<true>, dim3(T / 64, H), dim3(256), 0, s, p);
    } else {
        if (hpf) hipLaunchKernelGGL((g2::g2_wu_kernel<false, true>), dim3(T / 64, H), dim3(256), 0, s, p);
        else     hipLaunchKernelGGL(g2::g2_wu_kernel<false>, dim3(T / 64, H), dim3(256), 0, s, p);
    }""")
f.write_text(s)
print("P-84 APPLIED")
