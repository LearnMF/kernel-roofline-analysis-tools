"""kra P-85 (bitwise, dispatch only): dhu's 4-wave form (v8b4) window shrinks from 96..256 CTAs to 96..127
(KDA_G2_DHU4_MAX_CTAS).  Re-measured on P-82b (results/kda-ceiling-cards-20261016-p82 §3, b85a): the 8-wave form now
wins at 192 CTAs (H=48: dhu 1.050 8K, 1.065 16K, 1.094 32K, 1.145 64K) and is even at 128 (H=32: 1.013), while the
4-wave form still wins at 96 (H=24: 0.955 / 0.987 / 0.983 for the 8-wave form).  Both forms are bit-identical.
    python3 p85_dhu_window.py <dir holding g2_engine.cuh>"""
import sys
from pathlib import Path

f = Path(sys.argv[1]) / "g2_engine.cuh"
s = f.read_text()
old = "    int dhu4 = (ctas >= 96 && ctas <= 256) ? 1 : 0;"
new = ("    // kra P-85: re-measured on P-82b -- the 8-wave form wins from 128 CTAs on (H=48: 1.05-1.15x)\n"
       "    int dhu4 = (ctas >= 96 && ctas <= g2_env_long(\"KDA_G2_DHU4_MAX_CTAS\", 127)) ? 1 : 0;")
assert s.count(old) == 1
f.write_text(s.replace(old, new))
print("P-85 APPLIED")
