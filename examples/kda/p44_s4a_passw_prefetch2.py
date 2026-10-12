"""kra P-44 (A-class, production s4a): pass W's h / kgB tile prefetch two iterations deep.
SQTT (a81ded8, 16K/TP2): the per-kt s_waitcnt vmcnt(1) on the register-staged h tile (loaded one
step before it is written to LDS) is s4a's second stall, 7.7% of its wave-cycles.  Two register
slots by tile parity (even / odd) let tile kt+3 be issued while tile kt+2 is still in flight, so
every load has two loop bodies to land.  Same LDS double buffers, same values.
Apply after p43 (independent of it).      python3 p44_s4a_passw_prefetch2.py <tree>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wy.cuh"
t = P.read_text()
old = """            *reinterpret_cast<v4bh*>(&KGs[0][0] + 4 * tid) = ld8(kgsrc);            // tile 0
            v4bh kgreg = ld8(kgsrc + 1 * 256);                                      // tile 1
"""
new = """            *reinterpret_cast<v4bh*>(&KGs[0][0] + 4 * tid) = ld8(kgsrc);            // tile 0
            v4bh kgS1 = ld8(kgsrc + 1 * 256);                                       // tile 1 (odd slot)
            v4bh kgS0 = ld8(kgsrc + 2 * 256);                                       // tile 2 (even slot) kra P-44
"""
assert t.count(old) == 1, t.count(old); t = t.replace(old, new)
old = """            f4 hreg = *reinterpret_cast<const f4*>(hsrc);                       // block 0
            *reinterpret_cast<f4*>(hs + 8 * tid) = hreg;
            hreg = *reinterpret_cast<const f4*>(hsrc + 2048);                   // block 1
            lds_barrier();                                                      // block 0 visible
"""
new = """            *reinterpret_cast<f4*>(hs + 8 * tid) = *reinterpret_cast<const f4*>(hsrc);   // block 0
            f4 hS1 = *reinterpret_cast<const f4*>(hsrc + 1 * 2048);            // block 1 (odd slot)
            f4 hS0 = *reinterpret_cast<const f4*>(hsrc + 2 * 2048);            // block 2 (even slot) kra P-44
            lds_barrier();                                                      // block 0 visible
"""
assert t.count(old) == 1, t.count(old); t = t.replace(old, new)
old = """                if (kt < 7) {
                    // buffer (kt+1)&1 was last read in step kt-1, before the barrier ending kt-1
                    *reinterpret_cast<f4*>(hs + ((kt + 1) & 1) * 2048 + 8 * tid) = hreg;
                    *reinterpret_cast<v4bh*>(&KGs[(kt + 1) & 1][0] + 4 * tid) = kgreg;   // kra P-28
                    if (kt < 6) {
                        hreg = *reinterpret_cast<const f4*>(hsrc + (kt + 2) * 2048);
                        kgreg = ld8(kgsrc + (kt + 2) * 256);
                    }
                }
"""
new = """                if (kt < 7) {
                    // buffer (kt+1)&1 was last read in step kt-1, before the barrier ending kt-1
                    const bool odd = (kt + 1) & 1;                                // kra P-44: slot of tile kt+1
                    *reinterpret_cast<f4*>(hs + ((kt + 1) & 1) * 2048 + 8 * tid) = odd ? hS1 : hS0;
                    *reinterpret_cast<v4bh*>(&KGs[(kt + 1) & 1][0] + 4 * tid) = odd ? kgS1 : kgS0;
                    if (kt < 5) {                                                 // tile kt+3 -> the freed slot
                        const f4 hn = *reinterpret_cast<const f4*>(hsrc + (kt + 3) * 2048);
                        const v4bh kn = ld8(kgsrc + (kt + 3) * 256);
                        if (odd) { hS1 = hn; kgS1 = kn; } else { hS0 = hn; kgS0 = kn; }
                    }
                }
"""
assert t.count(old) == 1, t.count(old); t = t.replace(old, new)
P.write_text(t)
print("P-44 APPLIED")
