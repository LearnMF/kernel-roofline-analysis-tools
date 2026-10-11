"""kra P-28 (round 9): p3_wy_s4a's wave-invariant operand loads fetched ONCE per CTA via LDS (the
V5 / kra P-4 pattern that already serves the h block):
  * pass W's kgB B-fragments (4 x 8 B per lane per kt, identical for the 4 waves) -> per kt the CTA
    loads the 2 KB tile with ONE 8 B load per thread into a 2 x 2 KB double buffer next to the h
    buffers (prefetched with the h block, same barrier), every wave reads its fragments from LDS;
  * the dA stage's 16 scalar beta loads per lane (identical across waves) -> 64 masked floats
    staged once in the prologue (visible after its first barrier).
Values and MMAC order unchanged -> bit-identical.  LDS 26,112 -> 30,464 B (still 2 CTAs/CU).
Ablation upper bounds (8K/12 / 8K/96): kgB 1.088x / 1.008x, beta 1.050x / 1.013x on s4a.
Usage: python3 p28_s4a_dedupe.py <tree root>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wy.cuh"
t = P.read_text()


def sub(old, new):
    global t
    assert t.count(old) == 1, (old[:80], t.count(old))
    t = t.replace(old, new)


sub("""    __shared__ __align__(16) bf16 XT[kBT * kPT];    // dA-stage XC staging (dAT staging dropped)
    const int tid = threadIdx.x, r = tid >> 6, l = tid & 63, m = l & 15, q = l >> 4, pm = pi16(m);""",
    """    __shared__ __align__(16) bf16 XT[kBT * kPT];    // dA-stage XC staging (dAT staging dropped)
    __shared__ __align__(16) bf16 KGs[2][4 * 64 * 4];   // kra P-28: kgB tile of step kt (double buffer)
    __shared__ float BETAs[kBT];                        // kra P-28: masked beta of the chunk's rows
    const int tid = threadIdx.x, r = tid >> 6, l = tid & 63, m = l & 15, q = l >> 4, pm = pi16(m);""")
sub("""        const float beta_t = trok ? beta_[tr * H] : 0.f;
        v4bh aa[4];                                                         // A^T row tr = column tr of A""",
    """        const float beta_t = trok ? beta_[tr * H] : 0.f;
        if (tid < kBT) BETAs[tid] = g2_row_ok<M>(tid, cl) ? beta_[tid * H] : 0.f;   // kra P-28
        v4bh aa[4];                                                         // A^T row tr = column tr of A""")
sub("""                #pragma unroll
                for (int st = 0; st < 4; ++st) {
                    const v4bh kgb = ld8(kgB + ((st * 8 + kt) * 64 + l) * 4);
                    dA[st] = mmac(dwn, kgb, dA[st]);
                }
            };""",
    """                #pragma unroll
                for (int st = 0; st < 4; ++st) {
                    const v4bh kgb = ld8(&KGs[kt & 1][0] + (st * 64 + l) * 4);   // kra P-28
                    dA[st] = mmac(dwn, kgb, dA[st]);
                }
            };
            // kra P-28: thread tid carries fragment (st = tid>>6, lane = tid&63) of the kgB tile
            const bf16* kgsrc = kgB + (((tid >> 6) * 8) * 64 + (tid & 63)) * 4;
            *reinterpret_cast<v4bh*>(&KGs[0][0] + 4 * tid) = ld8(kgsrc);            // tile 0
            v4bh kgreg = ld8(kgsrc + 1 * 256);                                      // tile 1""")
sub("""                if (kt < 7) {
                    // buffer (kt+1)&1 was last read in step kt-1, before the barrier ending kt-1
                    *reinterpret_cast<f4*>(hs + ((kt + 1) & 1) * 2048 + 8 * tid) = hreg;
                    if (kt < 6) hreg = *reinterpret_cast<const f4*>(hsrc + (kt + 2) * 2048);
                }""",
    """                if (kt < 7) {
                    // buffer (kt+1)&1 was last read in step kt-1, before the barrier ending kt-1
                    *reinterpret_cast<f4*>(hs + ((kt + 1) & 1) * 2048 + 8 * tid) = hreg;
                    *reinterpret_cast<v4bh*>(&KGs[(kt + 1) & 1][0] + 4 * tid) = kgreg;   // kra P-28
                    if (kt < 6) {
                        hreg = *reinterpret_cast<const f4*>(hsrc + (kt + 2) * 2048);
                        kgreg = ld8(kgsrc + (kt + 2) * 256);
                    }
                }""")
sub("""                da[st][e] = bfr(tr > s ? dA[st][e] * (g2_row_ok<M>(s, cl) ? beta_[s * H] : 0.f) : 0.f);""",
    """                da[st][e] = bfr(tr > s ? dA[st][e] * BETAs[s] : 0.f);   // kra P-28: staged, same mask""")
P.write_text(t)
print("P-28 APPLIED")
