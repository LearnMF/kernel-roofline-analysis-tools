"""kra P-11: v8o4 MODE 1 (forward, o only) with paced loads: a scheduling fence after each
in-loop load group keeps the source spacing (P-9 mode 1 lost 4% because, without the
conservative waits, the compiler clustered the loads and VMEM issue stalls returned).
The fence (sched_barrier) emits no instruction; values unchanged.  Re-enables mode 1.
Usage: python3 p11_v8o4_pacing.py <g2 kernel dir>"""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "g2_fwd_v8.cuh"
t = p.read_text()
k0 = t.index("g2_fwd_h_v8o4_kernel(FwdOParams p) {")
k1 = t.index("\n}\n", k0)
body = t[k0:k1]
FENCE = "if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);   // kra P-11: keep load spacing"
for old in ("            loadQg(i); loadAq(i);                        // QgA/AqkA right after bar1",
            "        loadW(nxt);",
            "        loadU(nxt, 0, uf0); loadU(nxt, 1, uf1);",
            "        loadG(nxt, kb0, gl0); loadG(nxt, kb1, gl1);",
            "        loadK(nxt, kb0, kf0); loadK(nxt, kb1, kf1);"):
    assert body.count(old) == 1, (old, body.count(old))
    ind = old[:len(old) - len(old.lstrip())]
    body = body.replace(old, f"{ind}{FENCE}\n{old}\n{ind}{FENCE}")
t = t[:k0] + body + t[k1:]
old_l = """    int mode = 0;
    if (O == nullptr && Ob != nullptr && Vn && HbN && ANv) mode = 2;"""
new_l = """    int mode = 0;
    if (O == nullptr && Ob != nullptr && !Vn && !HbN && !ANv) mode = 1;   // kra P-11: paced mode 1
    else if (O == nullptr && Ob != nullptr && Vn && HbN && ANv) mode = 2;"""
assert t.count(old_l) == 1
t = t.replace(old_l, new_l)
old_c = "            case 2: hipLaunchKernelGGL((g2_fwd_h_v8o4_kernel<VL, 2>), grid, dim3(256), 0, s, p); break;"
assert t.count(old_c) == 1
t = t.replace(old_c, "            case 1: hipLaunchKernelGGL((g2_fwd_h_v8o4_kernel<VL, 1>), grid, dim3(256), 0, s, p); break;\n" + old_c)
p.write_text(t)
print("P-11 APPLIED")
