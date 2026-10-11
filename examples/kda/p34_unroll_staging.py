"""kra P-34 (round 11, bitwise-identical; found with SQTT): the cooperative staging loops written as
`for (int t = tid; t < N; t += 256)` are NOT unrolled by the compiler (runtime start), so each
iteration issues ONE global load and then `s_waitcnt vmcnt(0)` before its LDS stores -- N/256
serialized DRAM round trips.  SQTT on dav at 8K/12: the two such loops (Vn staging, do staging; 8
iterations) are 23.2% + 21.7% of ALL wave-cycles.  Rewriting them with a constant trip count and
#pragma unroll lets every load of the loop be in flight before the first wait.  Same per-element
operations and order -> bit-identical.  Applied to the production-path staging loops of dav (2), wu
(sA, phase-2 v), s4a (Y4 A staging, P-28c A re-stage) and s4b (Y4 A staging).
Usage: python3 p34_unroll_staging.py <tree root>"""
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) / "csrc" / "g2"
SITES = {   # file: list of (loop header, trip count) -- header must be unique in the file
    "g2_dav.cuh": [("        for (int t = tid; t < 2048; t += 256) {\n            const int sb = t >> 6, c = t & 63;", 8),
                   ("        for (int t = tid; t < 2048; t += 256) {\n            const int j = t >> 5, vq = (t & 31) << 2;\n            const v4bh dv_", 8)],
    "g2_wu.cuh": [("        for (int t = tid; t < 1024; t += 256) {          // sA: 64x64 bf16 row-major", 4),
                  ("        for (int t = tid; t < 2048; t += 256) {\n            const int j = t >> 5, vq = (t & 31) << 2;\n            const size_t so = sbase + (size_t)j * rstride + vq;\n            const v4bh vv", 8)],
    "g2_wy.cuh": [("            for (int j = tid; j < 64 * 16; j += 256) {", 4)],   # s4a Y4, P-28c, s4b Y4 (3 sites)
}
# P-34a (merged form): s4a's Y4 A staging (the FIRST g2_wy.cuh site) stays rolled -- unrolled it
# measured mixed (s4a 1.069x at 8K/12 but 0.963x at 16K/48); P34_ALL=1 restores all 7 sites.
import os
SKIP_FIRST_WY = os.environ.get("P34_ALL") != "1"
total = 0
for f, sites in SITES.items():
    p = ROOT / f
    t = p.read_text()
    for hdr, trips in sites:
        m = re.match(r"(\s*)for \(int (\w+) = tid; \w+ < [^;]+; \w+ \+= 256\) \{(.*)", hdr, re.S)
        indent, var, rest = m.group(1), m.group(2), m.group(3)
        new = (f"{indent}#pragma unroll   // kra P-34: all loads in flight before the first wait\n"
               f"{indent}for (int it_ = 0; it_ < {trips}; ++it_) {{\n{indent}    const int {var} = tid + 256 * it_;{rest}")
        n = t.count(hdr)
        assert n >= 1, (f, hdr[:60])
        if f == "g2_wy.cuh" and SKIP_FIRST_WY:
            i0 = t.index(hdr) + len(hdr)          # keep the first occurrence (s4a Y4) as is
            t = t[:i0] + t[i0:].replace(hdr, new)
            n -= 1
        else:
            t = t.replace(hdr, new)
        total += n
        print(f"{f}: {n} x '{hdr.strip().splitlines()[0][:60]}'")
    p.write_text(t)
print("P-34 APPLIED", total)
