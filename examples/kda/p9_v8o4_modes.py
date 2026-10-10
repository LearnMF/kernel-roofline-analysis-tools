"""kra P-9: v8o4 with compile-time output modes (D.waitcnt_conservative).
MODE 0 = generic (runtime null checks, the previous code); 1 = forward (Ob only);
2 = forward keep-state (Ob + HbN + Vn + ANv); 3 = backward recompute (HbN + Vn + ANv, no o).
The launcher picks the mode from the pointer pattern.  Math unchanged -> bit-identical.
Usage: python3 p9_v8o4_modes.py <g2 kernel dir>"""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "g2_fwd_v8.cuh"
t = p.read_text()
sig = "template <bool VL>\n__global__ void __launch_bounds__(256) g2_fwd_h_v8o4_kernel(FwdOParams p) {"
assert t.count(sig) == 1
k0 = t.index(sig)
k1 = t.index("\n}\n", k0)
body = t[k0:k1]
body = body.replace(sig, """template <bool VL, int MODE = 0>
__global__ void __launch_bounds__(256) g2_fwd_h_v8o4_kernel(FwdOParams p) {
    // kra P-9: compile-time outputs.  With runtime-optional VMEM ops in the loop the compiler's
    // waitcnt pass must count the path that skips them, so a vmcnt for an OLD load also forced
    // younger, actually-issued loads to land (v8o4 fwd: one GEMM1 wait = 15.7% of the step).
    // MODE 0 generic | 1 fwd (Ob) | 2 fwd keep-state (Ob+HbN+Vn+ANv) | 3 bwd recompute (HbN+Vn+ANv)
    const bool hasO  = MODE == 0 ? p.O != nullptr : false;
    const bool hasOb = MODE == 0 ? p.Ob != nullptr : MODE != 3;
    const bool hasBN = MODE == 0 ? p.HbN != nullptr : MODE != 1;
    const bool hasVn = MODE == 0 ? p.Vn != nullptr : MODE != 1;
    const bool hasAN = MODE == 0 ? p.ANv != nullptr : MODE != 1;""", 1)
for old, new, n in [
    ("if (p.HbN != nullptr) {", "if (hasBN) {", 1),
    ("if (p.O != nullptr || p.Ob != nullptr) {", "if (hasO || hasOb) {", 3),
    ("if (p.Vn != nullptr) {", "if (hasVn) {", 1),
    ("if (p.ANv != nullptr)", "if (hasAN)", 1),
    ("if (p.O != nullptr) {", "if (hasO) {", 1),
]:
    c = body.count(old)
    assert c == n, (old, c)
    body = body.replace(old, new)
rest = body.split("const bool hasAN = MODE == 0 ? p.ANv != nullptr : MODE != 1;")[1]
assert "nullptr" not in rest, "a runtime null check is left in the kernel body"
t = t[:k0] + body + t[k1:]
old_l = """    if (ctok != nullptr) {
        if (r1a) hipLaunchKernelGGL(g2_fwd_h_v8o4_kernel<true>, grid, dim3(256), 0, s, p);
        else     hipLaunchKernelGGL(g2_fwd_h_v8o_kernel<true>, grid, dim3(512), 0, s, p);
    } else {
        if (r1a) hipLaunchKernelGGL(g2_fwd_h_v8o4_kernel<false>, grid, dim3(256), 0, s, p);
        else     hipLaunchKernelGGL(g2_fwd_h_v8o_kernel<false>, grid, dim3(512), 0, s, p);
    }"""
new_l = """    // kra P-9: the production output sets get compile-time instances, others the generic one
    int mode = 0;
    if (O == nullptr && Ob != nullptr && !Vn && !HbN && !ANv) mode = 1;
    else if (O == nullptr && Ob != nullptr && Vn && HbN && ANv) mode = 2;
    else if (O == nullptr && Ob == nullptr && Vn && HbN && ANv) mode = 3;
    auto v8o4 = [&](auto vl) {
        constexpr bool VL = decltype(vl)::value;
        switch (mode) {
            case 1: hipLaunchKernelGGL((g2_fwd_h_v8o4_kernel<VL, 1>), grid, dim3(256), 0, s, p); break;
            case 2: hipLaunchKernelGGL((g2_fwd_h_v8o4_kernel<VL, 2>), grid, dim3(256), 0, s, p); break;
            case 3: hipLaunchKernelGGL((g2_fwd_h_v8o4_kernel<VL, 3>), grid, dim3(256), 0, s, p); break;
            default: hipLaunchKernelGGL((g2_fwd_h_v8o4_kernel<VL, 0>), grid, dim3(256), 0, s, p); break;
        }
    };
    if (ctok != nullptr) {
        if (r1a) v8o4(std::true_type{});
        else     hipLaunchKernelGGL(g2_fwd_h_v8o_kernel<true>, grid, dim3(512), 0, s, p);
    } else {
        if (r1a) v8o4(std::false_type{});
        else     hipLaunchKernelGGL(g2_fwd_h_v8o_kernel<false>, grid, dim3(512), 0, s, p);
    }"""
assert t.count(old_l) == 1
t = t.replace(old_l, new_l)
p.write_text(t)
print("P-9 APPLIED")
