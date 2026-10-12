"""kra P-47b (diagnostic, B-class): the fused tail's gate sigmoid with the hardware exp / rcp
(__expf, v_rcp_f32) instead of expf + IEEE division -- measures how much of the in-loop gate-backward
cost is the precise transcendental sequence.   python3 p47b_fast_sigmoid.py <tree with P-47>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wy.cuh"
t = P.read_text()
old = """                const float sg = 1.f / (1.f + expf(-(Ag * gg)));
                float bdg = suffix * (p.lower_bound * (sg * (1.f - sg)));"""
new = """                const float sg = __builtin_amdgcn_rcpf(1.f + __expf(-(Ag * gg)));   // kra P-47b
                float bdg = suffix * (p.lower_bound * (sg * (1.f - sg)));"""
assert t.count(old) == 1, t.count(old)
t = t.replace(old, new)
P.write_text(t)
print("P-47b APPLIED")
