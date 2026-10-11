"""kra P-28c (round 10, bitwise-identical): P-28b's dA-stage A-fragment staging (XT re-stage + 2
barriers, VMEM 16 -> 4 per lane) as a COMPILE-TIME variant of p3_wy_s4a (template AFS), selected only
for small head counts.  P-28b measured s4a 1.067x at 8K/12 but 0.938x / 0.926x at 8K/48 / 8K/96:
the A rows are H*64*2 B apart, so at large H the stride-RA fragment loads of the 4 waves are served
by L1/L2 and the two extra barriers cost more than the VMEM they save.  KDA_G2_S4A_AFS_MAXH (default
12; H=24 measured 1.036x at 8K but 0.94-0.97x at 32K-64K) sets the largest H that gets the variant.  Values unchanged -> bit-identical.
Usage: python3 p28c_s4a_afrag_lowh.py <tree root>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wy.cuh"
t = P.read_text()


def sub(old, new, count=1):
    global t
    assert t.count(old) == count, (old[:80], t.count(old))
    t = t.replace(old, new)


sub("""template <bool VL>
__global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(1, 3)))
p3_wy_s4a_kernel(WyS4AParams p) {""",
    """template <bool VL, bool AFS = false>   // kra P-28c: AFS = stage the dA-stage A block in XT
__global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(1, 3)))
p3_wy_s4a_kernel(WyS4AParams p) {""")
sub("""        f4 xc[4];
        #pragma unroll
        for (int ut = 0; ut < 4; ++ut) {
            xc[ut] = f4{0.f, 0.f, 0.f, 0.f};
            const bool aok = g2_row_ok<M>(16 * ut + pm, cl);
            #pragma unroll
            for (int st = 0; st < 4; ++st)
                xc[ut] = mmac(da[st], aok ? ld8(A_ + (16 * ut + pm) * RA + 16 * st + 4 * q)
                                          : v4bh{0, 0, 0, 0}, xc[ut]);
        }""",
    """        f4 xc[4];
        if constexpr (AFS) {   // kra P-28b/c: A block -> XT once per CTA (XT is free after pass W)
            short* As2 = reinterpret_cast<short*>(XT);
            for (int j = tid; j < 64 * 16; j += 256) {
                const int row = j >> 4, qc = j & 15;
                *reinterpret_cast<v4bh*>(As2 + row * kPT + 4 * qc) = g2_row_ok<M>(row, cl)
                    ? *reinterpret_cast<const v4bh*>(A_ + row * RA + 4 * qc) : v4bh{0, 0, 0, 0};
            }
            lds_barrier();                                                  // A block visible
            #pragma unroll
            for (int ut = 0; ut < 4; ++ut) {
                xc[ut] = f4{0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int st = 0; st < 4; ++st)
                    xc[ut] = mmac(da[st], ld8(XT + (16 * ut + pm) * kPT + 16 * st + 4 * q), xc[ut]);
            }
            lds_barrier();                                                  // A reads done -> XC
        } else {
        #pragma unroll
        for (int ut = 0; ut < 4; ++ut) {
            xc[ut] = f4{0.f, 0.f, 0.f, 0.f};
            const bool aok = g2_row_ok<M>(16 * ut + pm, cl);
            #pragma unroll
            for (int st = 0; st < 4; ++st)
                xc[ut] = mmac(da[st], aok ? ld8(A_ + (16 * ut + pm) * RA + 16 * st + 4 * q)
                                          : v4bh{0, 0, 0, 0}, xc[ut]);
        }
        }""")
sub("""        if (ctok) hipLaunchKernelGGL(p3_wy_s4a_kernel<true>, dim3(NT, H), dim3(256), 0, s, pa);
        else hipLaunchKernelGGL(p3_wy_s4a_kernel<false>, dim3(NT, H), dim3(256), 0, s, pa);""",
    """        int afs_maxh = 12;   // kra P-28c (H=24 measured mixed: 1.036x at 8K, 0.94-0.97x at 32K-64K)
        if (const char* e = getenv("KDA_G2_S4A_AFS_MAXH")) afs_maxh = atoi(e);
        const bool afs = H <= afs_maxh;
        if (ctok) {
            if (afs) hipLaunchKernelGGL((p3_wy_s4a_kernel<true, true>), dim3(NT, H), dim3(256), 0, s, pa);
            else     hipLaunchKernelGGL((p3_wy_s4a_kernel<true, false>), dim3(NT, H), dim3(256), 0, s, pa);
        } else {
            if (afs) hipLaunchKernelGGL((p3_wy_s4a_kernel<false, true>), dim3(NT, H), dim3(256), 0, s, pa);
            else     hipLaunchKernelGGL((p3_wy_s4a_kernel<false, false>), dim3(NT, H), dim3(256), 0, s, pa);
        }""")
P.write_text(t)
print("P-28c APPLIED")
