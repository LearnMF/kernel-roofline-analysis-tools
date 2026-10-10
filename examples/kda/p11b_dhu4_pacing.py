"""kra P-11b: dhu v8b4 production instance (MODE 1, P-10 lost 8% from load clustering) with the
same scheduling fences as P-11 around every in-loop load group; re-enables it in the launcher.
Usage: python3 p11b_dhu4_pacing.py <g2 kernel dir>"""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "g2_engine.cuh"
t = p.read_text()
k0 = t.index("g2_dhu_v8b4_kernel(BwdNParams p) {")
k1 = t.index("\n}\n", k0)
body = t[k0:k1]
FENCE = "if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);   // kra P-11b: keep load spacing"
for old in ("        loadDO(nxt, 0, dof0); loadDO(nxt, 1, dof1);",
            "        loadKA(nxt);",
            "        loadDV(nxt, 0, dvf0); loadDV(nxt, 1, dvf1);",
            "        loadG(nxt, kb0, gl0); loadG(nxt, kb1, gl1);",
            "        loadQW(nxt, kb0, qb0, wb0); loadQW(nxt, kb1, qb1, wb1);"):
    assert body.count(old) == 1, (old, body.count(old))
    body = body.replace(old, f"        {FENCE}\n{old}\n        {FENCE}")
t = t[:k0] + body + t[k1:]
old = """    if (dhu4) {
        // v8b4: the compile-time instance measured 8% SLOWER (kra P-10) -> generic only
        hipLaunchKernelGGL(g2_dhu_v8b4_kernel<0>, dim3(kK / kBV, nseg * H), dim3(256), 0, s, p);
    } else {"""
new = """    if (dhu4) {
        // v8b4: the unpaced compile-time instance measured 8% slower (kra P-10); the paced
        // one (sched fences around the load groups, kra P-11b) is used for the production set
        if (prod) hipLaunchKernelGGL(g2_dhu_v8b4_kernel<1>, dim3(kK / kBV, nseg * H), dim3(256), 0, s, p);
        else      hipLaunchKernelGGL(g2_dhu_v8b4_kernel<0>, dim3(kK / kBV, nseg * H), dim3(256), 0, s, p);
    } else {"""
assert t.count(old) == 1
t = t.replace(old, new)
p.write_text(t)
print("P-11b APPLIED")
