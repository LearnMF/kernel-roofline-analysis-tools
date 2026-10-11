"""kra P-22d (round 6): P-22c's linear aN(dv2) emit as a COMPILE-TIME variant (template LIN) of both
dhu kernels, selected by the host only in the latency-bound regime (CTAs <= 96: H=12 -> v8b,
H=24 -> v8b4); deep saturation keeps the scattered emit, whose stores pace the loads (R21).
KDA_G2_DHU_LIN=0/1 forces it.  Values unchanged -> bit-identical.
Usage: python3 p22d_dhu_lin_dispatch.py <tree root>"""
import re
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_engine.cuh"
t = P.read_text()

GATHER = """
        if (hasAN2 && LIN) {   // kra P-22c/d: aN(dv2) as linear 8 B stores gathered from the staged vB tile
            #pragma unroll
            for (int v1g = V0; v1g < V1; ++v1g) {
                v4bh an;
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    an[e] = reinterpret_cast<const short*>(vB)[(16 * v1g + 4 * (l >> 4) + e) * kPadT + 16 * r1 + (l & 15)];
                *reinterpret_cast<v4bh*>(p.ANv2 + ((size_t)i * p.H + bh) * 8192 + (8 * r1 + 2 * vs + v1g) * 256 + l * 4) = an;
            }
        }"""

for name, v0, v1 in (("g2_dhu_v8b_kernel", "v1", "v1 + 1"), ("g2_dhu_v8b4_kernel", "0", "2")):
    head = re.search(r"template <int MODE = 0>\s*\n__global__ void __launch_bounds__\((\d+)\) " + name + r"\(BwdNParams p\)", t)
    assert head, name
    t = t[:head.start()] + head.group(0).replace("template <int MODE = 0>", "template <int MODE = 0, bool LIN = false>") + t[head.end():]
    i = t.index(f"{name}(BwdNParams p)")
    j = t.index("\n}\n", i)
    seg = t[i:j]
    seg, n = re.subn(r"if \(hasAN2\)(?=[^\n]*\n\s*#pragma unroll\s*\n\s*for \(int e = 0; e < 4; \+\+e\)\s*\n\s*reinterpret_cast<short\*>\(p\.ANv2\))",
                     "if (hasAN2 && !LIN)", seg)
    assert n == 1, (name, n)
    bar = "lds_barrier();                                   // vB ready, dhB reads done"
    assert seg.count(bar) == 1, name
    seg = seg.replace(bar, bar + GATHER.replace("V0", v0).replace("V1", v1))
    t = t[:i] + seg + t[j:]
old = """    // kra P-10: the production output set gets the compile-time instance
    const bool prod = HbN != nullptr && ANv2 != nullptr && dv2 == nullptr;"""
assert t.count(old) == 1
t = t.replace(old, """    // kra P-22d: linear aN(dv2) emit only in the latency-bound regime (few CTAs); at deep saturation
    // the scattered stores pace the loads (R21).  KDA_G2_DHU_LIN=0/1 forces it.
    bool lin = ctas <= 96;
    if (const char* e = getenv("KDA_G2_DHU_LIN")) lin = atoi(e) != 0;
    // kra P-10: the production output set gets the compile-time instance
    const bool prod = HbN != nullptr && ANv2 != nullptr && dv2 == nullptr;""")
old = """        hipLaunchKernelGGL(g2_dhu_v8b4_kernel<0>, dim3(kK / kBV, nseg * H), dim3(256), 0, s, p);"""
assert t.count(old) == 1
t = t.replace(old, """        if (lin) hipLaunchKernelGGL((g2_dhu_v8b4_kernel<0, true>), dim3(kK / kBV, nseg * H), dim3(256), 0, s, p);
        else     hipLaunchKernelGGL((g2_dhu_v8b4_kernel<0, false>), dim3(kK / kBV, nseg * H), dim3(256), 0, s, p);""")
old = """        if (prod) hipLaunchKernelGGL(g2_dhu_v8b_kernel<1>, dim3(kK / kBV, nseg * H), dim3(512), 0, s, p);"""
assert t.count(old) == 1
t = t.replace(old, """        if (prod && lin) hipLaunchKernelGGL((g2_dhu_v8b_kernel<1, true>), dim3(kK / kBV, nseg * H), dim3(512), 0, s, p);
        else if (prod) hipLaunchKernelGGL((g2_dhu_v8b_kernel<1, false>), dim3(kK / kBV, nseg * H), dim3(512), 0, s, p);""")
P.write_text(t)
print("P-22d APPLIED")
