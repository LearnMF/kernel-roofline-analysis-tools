"""kra P-25 (round 8, numerics-relaxed by user decision, validated against the FP64 reference_kda.py):
every remaining exp2f() in the G2 kernels becomes the bare hardware exp2 (__builtin_amdgcn_exp2f,
one v_exp_f32).  exp2f lowers to a 6-instruction denormal-safe sequence (cmp, cndmask, add, v_exp,
cndmask, mul); for x >= -126 it adds 0 and multiplies by 1.0 -- bit-identical to the bare
instruction -- and for x < -126 it returns the denormal 2^x where the bare instruction flushes to 0.
P-2 already used the bare form where the argument range was PROVEN; the remaining sites take
in-chunk cumulative log2 gates (down to 64 x -7.2 = -461 under the -5 nat safe gate), so results
can differ -- by < 2^-126 relative to O(1) operands.  Sites: fwd_prep (VALU-bound, the target),
wu, the v8o/v8o4 and dhu state decays.  Code only (comments untouched).
P-25a (the merged form): fwd_prep + wu only (P25_FILES default).  The recurrence decays (v8o/v8o4,
dhu) keep exp2f: in the throttled deep-saturation regime (P-24) the shorter step let the loads
cluster (v8o4 MODE 4 -13..-18%, dhu -3% at H=96) -- R21 again.  P25_FILES=all restores P-25.
Usage: python3 p25_raw_exp2.py <tree root>"""
import os
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) / "csrc" / "g2"
total = 0
FILES = ("g2_fwd_prep.cuh", "g2_wu.cuh") + (("g2_fwd_v8.cuh", "g2_engine.cuh") if os.environ.get("P25_FILES") == "all" else ())
for f in FILES:
    p = ROOT / f
    out, n = [], 0
    for line in p.read_text().split("\n"):
        code, sep, comment = line.partition("//")
        new_code, k = re.subn(r"(?<![\w.])exp2f\(", "__builtin_amdgcn_exp2f(", code)
        n += k
        out.append(new_code + sep + comment)
    p.write_text("\n".join(out))
    print(f"{f}: {n} site(s)")
    total += n
assert total > 0
print("P-25 APPLIED", total)
