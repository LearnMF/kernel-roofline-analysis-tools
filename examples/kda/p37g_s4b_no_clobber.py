"""kra P-37g (round 12, s4b step 2e, A-class): drop the "memory" clobber from the pipelined s4b's
async-load and store asm so the scheduler may move LDS/VALU/MMAC work around them.

Why it is safe: an async stage load targets the stage consumed BEFORE the barrier that precedes it
(lds_barrier is a volatile asm with a memory clobber, and volatile asm statements keep their order),
and the stage is next read only after a later barrier; the buffer stores write global memory no
code in the kernel reads.  The m0 clobber stays.     python3 p37g_s4b_no_clobber.py <tree>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wy.cuh"
t = P.read_text()
o1 = ''':: "s"(m0), "v"(voff), "s"(srd) : "memory", "m0");'''
n1 = ''':: "s"(m0), "v"(voff), "s"(srd) : "m0");   // kra P-37g: no memory clobber'''
o2 = ''':: "v"(a), "v"(b), "v"(c), "v"(voff), "s"(rdq), "s"(rdk), "s"(rdg) : "memory");'''
n2 = ''':: "v"(a), "v"(b), "v"(c), "v"(voff), "s"(rdq), "s"(rdk), "s"(rdg));   // kra P-37g'''
assert t.count(o1) == 1 and t.count(o2) == 1, (t.count(o1), t.count(o2))
t = t.replace(o1, n1).replace(o2, n2)
P.write_text(t)
print("P-37g APPLIED")
