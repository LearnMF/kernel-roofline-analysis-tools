"""kra P-22c (round 6): dhu keeps emitting aN(dv2) -- P-22 showed that DROPPING its stores lets the
later loads cluster in the VMEM-issue-saturated recurrence (R21) -- but each lane's 4 (v8b) / 8
(v8b4) scattered 2 B stores per step become 1 / 2 linear 8 B stores (512 B per wave instruction).
The A-form quad (row 16 r1 + (l&15), cols 16 v1 + 4(l>>4) .. +3) is gathered from the vB tile the
kernel already stages, right after the existing "vB ready" barrier (4 scalar LDS reads per quad,
32 distinct banks per instruction).  The next step's vB writes sit behind the next step's first
barrier, so the extra reads are race-free.  Values unchanged -> bit-identical.
Usage: python3 p22c_dhu_an_linear.py <tree root>"""
import re
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_engine.cuh"
t = P.read_text()


def segment(name):
    i = t.index(f"{name}(BwdNParams p)")
    j = t.index("\n}\n", i)
    return i, j


GATHER = """
        if (hasAN2) {    // kra P-22c: aN(dv2) as linear 8 B stores, gathered from the staged vB tile
            #pragma unroll
            for (int v1g = {V0}; v1g < {V1}; ++v1g) {{
                v4bh an;
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    an[e] = reinterpret_cast<const short*>(vB)[(16 * v1g + 4 * (l >> 4) + e) * kPadT + 16 * r1 + (l & 15)];
                *reinterpret_cast<v4bh*>(p.ANv2 + ((size_t)i * p.H + bh) * 8192 + (8 * r1 + 2 * vs + v1g) * 256 + l * 4) = an;
            }}
        }"""

for name, v_range in (("g2_dhu_v8b_kernel", ("v1", "v1 + 1")), ("g2_dhu_v8b4_kernel", ("0", "2"))):
    i, j = segment(name)
    seg = t[i:j]
    # 1) drop the scattered per-element emit
    seg, n = re.subn(r"\n\s*if \(hasAN2\)[^\n]*\n\s*#pragma unroll\s*\n\s*for \(int e = 0; e < 4; \+\+e\)\s*\n"
                     r"\s*reinterpret_cast<short\*>\(p\.ANv2\)\[[^;]*;", "", seg)
    assert n == 1, (name, "scattered emit sites", n)
    # 2) linear emit right after the vB-ready barrier
    bar = "lds_barrier();                                   // vB ready, dhB reads done"
    assert seg.count(bar) == 1, (name, "barrier", seg.count(bar))
    seg = seg.replace(bar, bar + GATHER.replace("{V0}", v_range[0]).replace("{V1}", v_range[1])
                      .replace("{{", "{").replace("}}", "}"))
    t = t[:i] + seg + t[j:]
    print("patched", name)
P.write_text(t)
print("P-22c APPLIED")
