"""kra P-73a ABLATION (wrong numbers, timing only): upper bound of an s4b-native layout for g / k / q.

TA microbench (results/kda-bound-20261012): s4b's per-kt tile reads touch 16 rows per instruction -- g (16 B/lane,
16 rows x 64 B) costs 64 / 169 CU-cycles (L2 / HBM) and k, q (8 B/lane, 16 rows x 32 B) 64 / 85, against 17-22 for
row-contiguous 512 B-1 KB.  This variant keeps every instruction and byte count but makes those three loads
contiguous (each wave reads a 1 KB / 512 B contiguous run inside the same chunk-head region), so the s4b time delta
bounds what re-laying g / k / q out for s4b could buy.  Numbers are wrong -- never gate this.
    python3 p73a_s4b_ablate.py <tree> [g|kq|all]"""
import sys
from pathlib import Path

f = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wy.cuh"
which = sys.argv[2] if len(sys.argv) > 2 else "all"
s = f.read_text()
head, rest = s.split("__device__ __forceinline__ void s4b_tail(", 1)
body, tail = rest.split("p3_wy_s4b_kernel(WyS4BParams p) {", 1)
old_g = "        const f4 gv = ld16f(g_ + rog + kc);"
old_k = "        const v4bh kk = rok ? ld8(k_ + ro + kc) : v4bh{0, 0, 0, 0};"
old_q = "        const v4bh qq = rok ? ld8(q_ + ro + kc) : v4bh{0, 0, 0, 0};"
for o in (old_g, old_k, old_q):
    assert body.count(o) == 1, o
if which in ("g", "all"):
    body = body.replace(old_g, "        const f4 gv = ld16f(g_ + (16 * A + kt) * 256 + 4 * l);   // P-73a ABLATION: contiguous 1 KB")
if which in ("kq", "all"):
    body = body.replace(old_k, "        const v4bh kk = ld8(k_ + (16 * A + kt) * 256 + 4 * l);   // P-73a ABLATION: contiguous 512 B")
    body = body.replace(old_q, "        const v4bh qq = ld8(q_ + (16 * A + kt) * 256 + 4 * l);   // P-73a ABLATION")
f.write_text(head + "__device__ __forceinline__ void s4b_tail(" + body + "p3_wy_s4b_kernel(WyS4BParams p) {" + tail)
print("P-73a ABLATION APPLIED:", which)
