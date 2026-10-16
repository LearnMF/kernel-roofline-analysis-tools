"""kra P-79c ABLATION (WRONG numbers, timing only): s4b's two per-kt LDS barriers removed -- upper bound of what
software-pipelining the kt loop (two kt in flight, double-buffered tables) could recover at 1 CTA/CU.  Selected by
KDA_G2_S4B_OT=3 on top of p78a + p79.     python3 p79c_s4b_nobar.py <tree>"""
import sys
from pathlib import Path

f = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wy.cuh"
s = f.read_text()
head, rest = s.split("__device__ __forceinline__ void s4b_tail(", 1)
body, tail = rest.split("p3_wy_s4b_kernel(WyS4BParams p) {", 1)
o1 = """            if (kt + 1 < 8) dwvN = ld8(dwTg + (kc + 16 + dwr) * kBT + 4 * dwc);   // kra P-79: kt+1
        lds_barrier();"""
assert body.count(o1) == 1
body = body.replace(o1, """            if (kt + 1 < 8) dwvN = ld8(dwTg + (kc + 16 + dwr) * kBT + 4 * dwc);   // kra P-79: kt+1
        if constexpr (OT != 3) lds_barrier(); else asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");   // P-79c""")
o2 = """        lds_barrier();                             // table WAR: kt's reads done before kt+1 writes
    }"""
assert body.count(o2) == 1
body = body.replace(o2, """        if constexpr (OT != 3) lds_barrier();      // table WAR: kt's reads done before kt+1 writes   (P-79c: off)
    }""")
o0 = "        if constexpr (OT == 0) {   // production order"
assert body.count(o0) == 1
body = body.replace(o0, "        if constexpr (OT == 0 || OT == 3) {   // production order (P-79c: OT 3 too)")
s = head + "__device__ __forceinline__ void s4b_tail(" + body + "p3_wy_s4b_kernel(WyS4BParams p) {" + tail
o3 = """        else if (ot == 2) hipLaunchKernelGGL((p3_wy_s4b_kernel<false, 2>), dim3(NT, H), dim3(256), padl, s, pb);"""
assert s.count(o3) == 1
s = s.replace(o3, o3 + """
        else if (ot == 3) hipLaunchKernelGGL((p3_wy_s4b_kernel<false, 3>), dim3(NT, H), dim3(256), padl, s, pb);""")
f.write_text(s)
print("P-79c APPLIED")
