"""kra P-63b probe (on top of P-63): with BP the K-operand prefetch is issued AFTER write_h + the bN emission.
P-63's SQTT (cp_p63_d3): the next step's K loads (issued before write_h) became the critical path -- 1079 of 2473
cycles/step waiting on one of them.  Issue order only; values unchanged.
    python3 p63b_order.py <tree>"""
import sys
from pathlib import Path

f = Path(sys.argv[1]) / "csrc" / "g2" / "g2_fwd_v8.cuh"
s = f.read_text()
head, rest = s.split("g2_fwd_h_v8o4_kernel(FwdOParams p) {", 1)
body, tail = rest.split("// kra P-49: g2_fwd_h_v8o4_kernel with NVH", 1)
old = """        if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);   // kra P-11: keep load spacing
        loadK(nxt, kb0, kf0); loadK(nxt, kb1, kf1);
        if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);   // kra P-11: keep load spacing
        write_h(kb0, ht0, i + 1 < c1 ? i + 1 : -1);   // kra P-63: h_{i+1} -> bN(h_{i+1})
        write_h(kb1, ht1, i + 1 < c1 ? i + 1 : -1);
"""
new = """        if constexpr (!BP) {
            if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);   // kra P-11: keep load spacing
            loadK(nxt, kb0, kf0); loadK(nxt, kb1, kf1);
            if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);   // kra P-11: keep load spacing
        }
        write_h(kb0, ht0, i + 1 < c1 ? i + 1 : -1);   // kra P-63: h_{i+1} -> bN(h_{i+1})
        write_h(kb1, ht1, i + 1 < c1 ? i + 1 : -1);
        if constexpr (BP) {   // kra P-63b: K prefetch after the emission
            __builtin_amdgcn_sched_barrier(0);
            loadK(nxt, kb0, kf0); loadK(nxt, kb1, kf1);
        }
"""
assert body.count(old) == 1
body = body.replace(old, new)
f.write_text(head + "g2_fwd_h_v8o4_kernel(FwdOParams p) {" + body + "// kra P-49: g2_fwd_h_v8o4_kernel with NVH" + tail)
print("P-63b APPLIED")
