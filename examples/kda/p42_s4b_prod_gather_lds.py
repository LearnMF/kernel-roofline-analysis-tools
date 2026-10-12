"""kra P-42 (A-class, production s4b at 2 CTAs/CU): the prologue's dA / dAqk operands via LDS tiles.

waterlevel (a81ded8, 16K/TP2): wy_s4_split = 8.9 ms at 36% of its interface bound, VMEM-issue 73%.
s4b's prologue reads the dA handoff and dAqk 64x64 fp32 tiles as rows (dak, aqA: 16 B) AND as
transposed columns (akB, aqB: 2 x 4 x (4-A) scalar 4 B loads per lane, 32 for wave 0).  Here each
tile is fetched once with coalesced 16 B loads (4 per thread) into the 30 KB the tables + stage
buffers leave free before the loop, in two rounds (dA, then dAqk: 16 KB each), and every operand
is read from LDS.  Masked dAqk rows are stored as zeros (same values as the rok / row_ok selects).
Values unchanged.          python3 p42_s4b_prod_gather_lds.py <tree>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wy.cuh"
t = P.read_text()
head = "__device__ __forceinline__ void s4b_tail(const WyS4BParams& p, S4BSmem& sm, const v4bh (&aa)[4],"
i = t.index(head)
j = t.index("    const size_t cQ = g2_nat_blk(i, hh, H, 0, true, 8 * 8 * 256);", i)
old = t[t.index("    f4 dak[4];\n", i):j]
assert "akB[b - A][x] = keep ? dAg[j * RA + row] : 0.f;" in old, "unexpected prologue"
new = """    // kra P-42: dA / dAqk tiles through LDS (the tables + stage buffers are free before the loop;
    // 30,720 contiguous bytes from TA -> one 16 KB tile per round)
    float* const lt = reinterpret_cast<float*>(&sm.TA[0]);
    const int tid_ = threadIdx.x;
    f4 dak[4];
    f4 aqA[A + 1], aqB[4 - A], akB[4 - A];
    {
        f4 tv[4];
        #pragma unroll
        for (int it_ = 0; it_ < 4; ++it_) {                    // round 1: dA (chunk-padded, all rows)
            const int u = it_ * 256 + tid_, r_ = u >> 4, c_ = u & 15;
            tv[it_] = ld16f(dAg + r_ * RA + 4 * c_);
        }
        #pragma unroll
        for (int it_ = 0; it_ < 4; ++it_)
            *reinterpret_cast<f4*>(&lt[(it_ * 256 + tid_) * 4]) = tv[it_];
        lds_barrier();
        #pragma unroll
        for (int ut = 0; ut < 4; ++ut) dak[ut] = *reinterpret_cast<const f4*>(&lt[row * 64 + 16 * ut + 4 * q]);
        #pragma unroll
        for (int b = A; b < 4; ++b)
            #pragma unroll
            for (int x = 0; x < 4; ++x) {
                const int j_ = 16 * b + 4 * q + x;
                const bool keep = b > A || 4 * q + x >= m;
                akB[b - A][x] = keep ? lt[j_ * 64 + row] : 0.f;   // dAT[row][j] == dA[j][row]
            }
        #pragma unroll
        for (int it_ = 0; it_ < 4; ++it_) {                    // round 2: dAqk (masked rows -> 0)
            const int u = it_ * 256 + tid_, r_ = u >> 4, c_ = u & 15;
            tv[it_] = g2_row_ok<M>(r_, cl) ? ld16f(aq_ + r_ * RA + 4 * c_) : f4{0.f, 0.f, 0.f, 0.f};
        }
        lds_barrier();                                         // round-1 reads done
        #pragma unroll
        for (int it_ = 0; it_ < 4; ++it_)
            *reinterpret_cast<f4*>(&lt[(it_ * 256 + tid_) * 4]) = tv[it_];
        lds_barrier();
        #pragma unroll
        for (int b = 0; b <= A; ++b) {
            aqA[b] = rok ? *reinterpret_cast<const f4*>(&lt[row * 64 + 16 * b + 4 * q]) : f4{0.f, 0.f, 0.f, 0.f};
            if (b == A) {
                #pragma unroll
                for (int x = 0; x < 4; ++x)
                    if (4 * q + x > m) aqA[b][x] = 0.f;
            }
        }
        #pragma unroll
        for (int b = A; b < 4; ++b)
            #pragma unroll
            for (int x = 0; x < 4; ++x) {
                const int j_ = 16 * b + 4 * q + x;
                const bool keep = b > A || 4 * q + x >= m;
                aqB[b - A][x] = (keep && g2_row_ok<M>(j_, cl)) ? lt[j_ * 64 + row] : 0.f;
            }
        lds_barrier();                                         // tiles consumed: stage buffers free
    }
"""
t = t[:t.index("    f4 dak[4];\n", i)] + new + t[j:]
P.write_text(t)
print("P-42 APPLIED")
