"""kra P-10: dhu v8b / v8b4 with a compile-time instance for the production output set
(HbN + ANv2, no dv2, no dh); other sets keep the generic (runtime-checked) instance.
Same mechanism as P-9 (D.waitcnt_conservative).  Math unchanged -> bit-identical.
Usage: python3 p10_dhu_modes.py <g2 kernel dir>"""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "g2_engine.cuh"
t = p.read_text()
FLAGS = """    // kra P-10: MODE 1 = production output set (HbN + ANv2; dv2 and dh null) as compile-time
    // constants -- runtime-optional VMEM ops make the waitcnt pass count the skip path.
    const bool hasBN = MODE == 0 ? p.HbN != nullptr : true;
    const bool hasDH = MODE == 0 ? p.dh != nullptr : false;
    const bool hasDV2 = MODE == 0 ? p.dv2 != nullptr : false;
    const bool hasAN2 = MODE == 0 ? p.ANv2 != nullptr : true;"""
for sig, new_sig in (
        ('extern "C" __global__ void __launch_bounds__(512) g2_dhu_v8b_kernel(BwdNParams p) {',
         'template <int MODE = 0>\n__global__ void __launch_bounds__(512) g2_dhu_v8b_kernel(BwdNParams p) {'),
        ('__global__ void __launch_bounds__(256) g2_dhu_v8b4_kernel(BwdNParams p) {',
         'template <int MODE = 0>\n__global__ void __launch_bounds__(256) g2_dhu_v8b4_kernel(BwdNParams p) {')):
    assert t.count(sig) == 1, sig
    k0 = t.index(sig)
    k1 = t.index("\n}\n", k0)
    body = t[k0:k1].replace(sig, new_sig + "\n" + FLAGS, 1)
    for old, new in (("if (p.HbN != nullptr)", "if (hasBN)"), ("if (p.dh != nullptr)", "if (hasDH)"),
                     ("if (p.dv2 != nullptr)", "if (hasDV2)"), ("if (p.ANv2 != nullptr)", "if (hasAN2)")):
        assert old in body, (sig[:40], old)
        body = body.replace(old, new)
    rest = body.split("const bool hasAN2 = MODE == 0 ? p.ANv2 != nullptr : true;")[1]
    assert "nullptr" not in rest, (sig[:40], "runtime null check left")
    t = t[:k0] + body + t[k1:]
old = """    if (dhu4)
        hipLaunchKernelGGL(g2_dhu_v8b4_kernel, dim3(kK / kBV, nseg * H), dim3(256), 0, s, p);
    else
        hipLaunchKernelGGL(g2_dhu_v8b_kernel, dim3(kK / kBV, nseg * H), dim3(512), 0, s, p);"""
new = """    // kra P-10: the production output set gets the compile-time instance
    const bool prod = HbN != nullptr && ANv2 != nullptr && dv2 == nullptr;
    if (dhu4) {
        if (prod) hipLaunchKernelGGL(g2_dhu_v8b4_kernel<1>, dim3(kK / kBV, nseg * H), dim3(256), 0, s, p);
        else      hipLaunchKernelGGL(g2_dhu_v8b4_kernel<0>, dim3(kK / kBV, nseg * H), dim3(256), 0, s, p);
    } else {
        if (prod) hipLaunchKernelGGL(g2_dhu_v8b_kernel<1>, dim3(kK / kBV, nseg * H), dim3(512), 0, s, p);
        else      hipLaunchKernelGGL(g2_dhu_v8b_kernel<0>, dim3(kK / kBV, nseg * H), dim3(512), 0, s, p);
    }"""
assert t.count(old) == 1
t = t.replace(old, new)
p.write_text(t)
print("P-10 APPLIED")
