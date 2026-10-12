"""kra P-43 (A-class, production s4a): the Y4 A-block staging loop unrolled with all four loads in
flight before the LDS writes.  SQTT (a81ded8, 16K/TP2): the loop's per-iteration load ->
s_waitcnt vmcnt(0) -> ds_write chain is s4a's largest stall, 10.9% of its wave-cycles (round 11
left this one loop rolled after mixed 8K results).  Values unchanged.
python3 p43_s4a_y4_unroll.py <tree>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wy.cuh"
t = P.read_text()
old = """            short* As = reinterpret_cast<short*>(S);
            for (int j = tid; j < 64 * 16; j += 256) {
                const int row = j >> 4, qc = j & 15;
                // VL: rows past the chunk's end are NOT in the packed buffer -- never load them
                // (their aa entries are masked below anyway).
                *reinterpret_cast<v4bh*>(As + row * 64 + 4 * qc) = g2_row_ok<M>(row, cl)
                    ? *reinterpret_cast<const v4bh*>(A_ + row * RA + 4 * qc) : v4bh{0, 0, 0, 0};
            }
"""
new = """            short* As = reinterpret_cast<short*>(S);
            v4bh av_[4];                                   // kra P-43: all four loads in flight
            #pragma unroll
            for (int it_ = 0; it_ < 4; ++it_) {
                const int j = tid + 256 * it_, row = j >> 4, qc = j & 15;
                // VL: rows past the chunk's end are NOT in the packed buffer -- never load them
                // (their aa entries are masked below anyway).
                av_[it_] = g2_row_ok<M>(row, cl)
                    ? *reinterpret_cast<const v4bh*>(A_ + row * RA + 4 * qc) : v4bh{0, 0, 0, 0};
            }
            #pragma unroll
            for (int it_ = 0; it_ < 4; ++it_) {
                const int j = tid + 256 * it_, row = j >> 4, qc = j & 15;
                *reinterpret_cast<v4bh*>(As + row * 64 + 4 * qc) = av_[it_];
            }
"""
assert t.count(old) == 1, t.count(old)
t = t.replace(old, new)
P.write_text(t)
print("P-43 APPLIED")
