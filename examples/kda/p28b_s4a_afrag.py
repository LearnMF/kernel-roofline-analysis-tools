"""kra P-28b (round 9, on top of P-28): s4a's dA stage reads its A B-fragments (16 x 8 B per lane,
stride-RA rows, identical for the 4 waves) straight from global.  XT is free between the end of pass
W and the XC write, so the CTA re-stages the 64x64 A block there with 4 coalesced 8 B loads per
thread (pitch kPT: conflict-free fragment reads; rows past the chunk end staged as 0 = the same mask
as before), every wave reads its fragments from LDS, and one extra barrier orders those reads
before the XC write.  Values and MMAC order unchanged -> bit-identical.  VMEM per lane 16 -> 4.
Usage: python3 p28b_s4a_afrag.py <tree root>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wy.cuh"
t = P.read_text()


def sub(old, new):
    global t
    assert t.count(old) == 1, (old[:80], t.count(old))
    t = t.replace(old, new)


sub("""        f4 xc[4];
        #pragma unroll
        for (int ut = 0; ut < 4; ++ut) {
            xc[ut] = f4{0.f, 0.f, 0.f, 0.f};
            const bool aok = g2_row_ok<M>(16 * ut + pm, cl);
            #pragma unroll
            for (int st = 0; st < 4; ++st)
                xc[ut] = mmac(da[st], aok ? ld8(A_ + (16 * ut + pm) * RA + 16 * st + 4 * q)
                                          : v4bh{0, 0, 0, 0}, xc[ut]);
        }
        #pragma unroll
        for (int ut = 0; ut < 4; ++ut)""",
    """        f4 xc[4];
        {   // kra P-28b: A block -> XT once per CTA (XT is free after pass W's last barrier)
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
        }
        #pragma unroll
        for (int ut = 0; ut < 4; ++ut)""")
P.write_text(t)
print("P-28b APPLIED")
