"""kra P-64 probe variants (bitwise, issue order only) for the PF (TP8 MODE 1) recurrence instance.

SQTT of the production step after P-62 (cp_p62_d2): the next step's K prefetch (4 dwordx4, issued after GEMM2) waits
266 cycles to ISSUE -- right before it the step issued QgA(i+1) (4 loads), the two o stores and AqkA(i+1) (2 loads),
so the K loads queue behind ~12 VMEM ops in the TA, and write_h (behind them in program order) waits with them.
    A: AqkA(i+1) moves from after the o pass to after the K prefetch (end of step)
    B: the K prefetch moves after write_h (write_h no longer queues behind it)
    C: AqkA(i) goes back to the start of its own step (right after barrier A; QgA stays one step ahead)
    python3 p64_vmem_order.py <tree> A|B|C"""
import sys
from pathlib import Path

f = Path(sys.argv[1]) / "csrc" / "g2" / "g2_fwd_v8.cuh"
var = sys.argv[2]
s = f.read_text()
head, rest = s.split("g2_fwd_h_v8o4_kernel(FwdOParams p) {", 1)
body, tail = rest.split("// kra P-49: g2_fwd_h_v8o4_kernel with NVH", 1)


def sub(b, old, new):
    assert b.count(old) == 1, (old[:80], b.count(old))
    return b.replace(old, new)


AQ = ("        if constexpr (PF) { if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0); loadAq(nxt); "
      "if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0); }   // kra P-60: aqf's last use was the o pass\n")
KW = """        if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);   // kra P-11: keep load spacing
        loadK(nxt, kb0, kf0); loadK(nxt, kb1, kf1);
        if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);   // kra P-11: keep load spacing
        write_h(kb0, ht0);
        write_h(kb1, ht1);
"""
if var == "A":
    body = sub(body, AQ, "")
    body = sub(body, KW, KW.replace("        write_h(kb0, ht0);\n",
                                    "        if constexpr (PF) loadAq(nxt);   // kra P-64A\n        write_h(kb0, ht0);\n"))
elif var == "B":
    body = sub(body, KW, """        if constexpr (!PF) {
            if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);   // kra P-11: keep load spacing
            loadK(nxt, kb0, kf0); loadK(nxt, kb1, kf1);
            if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);   // kra P-11: keep load spacing
        }
        write_h(kb0, ht0);
        write_h(kb1, ht1);
        if constexpr (PF) {   // kra P-64B: K prefetch after write_h
            __builtin_amdgcn_sched_barrier(0);
            loadK(nxt, kb0, kf0); loadK(nxt, kb1, kf1);
        }
""")
elif var == "C":
    body = sub(body, AQ, "")
    body = sub(body, "    if constexpr (PF) { loadQg(c0); loadAq(c0); }   // kra P-60: QgA/AqkA one step ahead",
               "    if constexpr (PF) { loadQg(c0); }   // kra P-60 / P-64C: QgA one step ahead")
    body = sub(body, """            loadQg(i); loadAq(i);                        // QgA/AqkA right after bar1
            if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);   // kra P-11: keep load spacing
        }""", """            loadQg(i); loadAq(i);                        // QgA/AqkA right after bar1
            if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);   // kra P-11: keep load spacing
        }
        if constexpr (PF) {   // kra P-64C: AqkA(i) at the start of its own step
            __builtin_amdgcn_sched_barrier(0);
            loadAq(i);
            __builtin_amdgcn_sched_barrier(0);
        }""")
else:
    raise SystemExit("variant A|B|C")
f.write_text(head + "g2_fwd_h_v8o4_kernel(FwdOParams p) {" + body + "// kra P-49: g2_fwd_h_v8o4_kernel with NVH" + tail)
print("P-64" + var + " APPLIED")
