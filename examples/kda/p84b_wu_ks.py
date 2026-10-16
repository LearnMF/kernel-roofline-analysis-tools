"""kra P-84b probe (bitwise): wu phase 1 issues all its k / g loads before the first use (template bool KS,
KDA_G2_WU_KS=1).

P-84 (VNEW bN prefetch) left the wait in place: SQTT with HPF=1 still shows the 8-per-wave vmcnt(1) at 14.7% -- it is
phase 1, not VNEW.  Phase 1's unrolled loop loads k (8 B) and g (16 B fp32) per quad and, in the same iteration,
stores kgB / KgA to global; the stores may alias the next iteration's loads (no __restrict__), so the compiler keeps
load(it+1) behind store(it) and each quad pays the full load latency.  KS issues the 8 k quads into kgr[] and the 8 g
quads into egr[] first (the arrays already carried to phase 3 -- no new registers), then the compute loop reads them
and overwrites the slots with gkv / exp2f(g) as before.  Values unchanged.
    python3 p84b_wu_ks.py <dir holding g2_wu.cuh>"""
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
    """template <bool VL, bool KS = false>   // kra P-84b: KS
__global__ void __launch_bounds__(256) g2_wu_kernel(WuParams p_) {""")
sub("""        v4bh kgr[8];                                     // kra P-19: kg quads carried to phase 3
        f4 egr[8];                                       // kra P-19: exp2f(g) carried to phase 3
        #pragma unroll
        for (int it = 0; it < 8; ++it) {
            const int t = tid + 256 * it;
            const int j = t >> 5, vq = (t & 31) << 2;
            const size_t so = sbase + (size_t)j * rstride + vq;
            const v4bh kv = g2_row_ok<M>(j, cl) ? *reinterpret_cast<const v4bh*>(p.k + so)
                                                 : v4bh{0, 0, 0, 0};
            const f4 gf = ld16f(p.g + sbase + (size_t)g2_row_g<M>(j, cl) * rstride + vq);""",
    """        v4bh kgr[8];                                     // kra P-19: kg quads carried to phase 3
        f4 egr[8];                                       // kra P-19: exp2f(g) carried to phase 3
        if constexpr (KS) {   // kra P-84b: every k / g quad in flight before the first use (the kgB / KgA
            #pragma unroll    // stores of an iteration would otherwise pin the next iteration's loads)
            for (int it = 0; it < 8; ++it) {
                const int t = tid + 256 * it;
                const int j = t >> 5, vq = (t & 31) << 2;
                const size_t so = sbase + (size_t)j * rstride + vq;
                kgr[it] = g2_row_ok<M>(j, cl) ? *reinterpret_cast<const v4bh*>(p.k + so) : v4bh{0, 0, 0, 0};
                egr[it] = ld16f(p.g + sbase + (size_t)g2_row_g<M>(j, cl) * rstride + vq);
            }
        }
        #pragma unroll
        for (int it = 0; it < 8; ++it) {
            const int t = tid + 256 * it;
            const int j = t >> 5, vq = (t & 31) << 2;
            const size_t so = sbase + (size_t)j * rstride + vq;
            const v4bh kv = KS ? kgr[it] : (g2_row_ok<M>(j, cl) ? *reinterpret_cast<const v4bh*>(p.k + so)
                                                 : v4bh{0, 0, 0, 0});
            const f4 gf = KS ? egr[it] : ld16f(p.g + sbase + (size_t)g2_row_g<M>(j, cl) * rstride + vq);""")
sub("""    if (ctok)
        hipLaunchKernelGGL(g2::g2_wu_kernel<true>, dim3(T / 64, H), dim3(256), 0, s, p);
    else
        hipLaunchKernelGGL(g2::g2_wu_kernel<false>, dim3(T / 64, H), dim3(256), 0, s, p);""",
    """    const bool ks = g2::g2_env_long("KDA_G2_WU_KS", 0) != 0;   // kra P-84b probe
    if (ctok) {
        if (ks) hipLaunchKernelGGL((g2::g2_wu_kernel<true, true>), dim3(T / 64, H), dim3(256), 0, s, p);
        else    hipLaunchKernelGGL(g2::g2_wu_kernel<true>, dim3(T / 64, H), dim3(256), 0, s, p);
    } else {
        if (ks) hipLaunchKernelGGL((g2::g2_wu_kernel<false, true>), dim3(T / 64, H), dim3(256), 0, s, p);
        else    hipLaunchKernelGGL(g2::g2_wu_kernel<false>, dim3(T / 64, H), dim3(256), 0, s, p);
    }""")
f.write_text(s)
print("P-84b APPLIED")
