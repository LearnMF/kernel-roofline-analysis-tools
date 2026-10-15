"""kra P-63d probe (on top of P-63 + P-63c): the selector-MMAC outputs are exact bf16 values held in fp32 (low 16 bits
zero), so bf16 = the high half -- one shift instead of bf_rne's bfe + add3 (identical bits for exact values).
    python3 p63d_trunc.py <tree>"""
import sys
from pathlib import Path

f = Path(sys.argv[1]) / "csrc" / "g2" / "g2_fwd_v8.cuh"
s = f.read_text()
old = "                    for (int e = 0; e < 4; ++e) bnv[kb == kb0 ? 0 : 1][vt][e] = bf_rne(c[e]);"
assert s.count(old) == 1
s = s.replace(old, "                    for (int e = 0; e < 4; ++e)   // kra P-63d: exact bf16 values -> high half\n"
                   "                        bnv[kb == kb0 ? 0 : 1][vt][e] = (short)(__float_as_uint(c[e]) >> 16);")
f.write_text(s)
print("P-63d APPLIED")
