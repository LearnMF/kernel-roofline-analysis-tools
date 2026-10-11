"""kra P-35 (round 11, bitwise-identical; found with SQTT + LLVM IR): every memory op of g2_wu_kernel
(dense AND varlen, 82 / 166) and of p3_wy_s4a_kernel<VL=true> (187) is FLAT, never global (in the
P-34 build; production had only s4a VL flat -- P-34's unrolling flipped wu and dav VL too, i.e. the
inference is fragile, so the fix stamps all three kernels).  In the
IR the byref kernel-argument struct is addrspacecast to a generic pointer and the fields are loaded
through it, so AMDGPUPromoteKernelArguments never sees a kernarg load and the pointers stay generic.
Flat ops also count on lgkmcnt (LDS waits drain global loads and vice versa) and cannot use
SGPR-base addressing.  dav / s4b load the fields straight from addrspace(4) and are global.
Fix: a field-wise local copy at kernel scope in which every pointer is stamped global
(generic -> addrspace(1) -> generic, g2_glb): the global origin then lives in the SSA value itself and
InferAddressSpaces turns every access into global_* whatever memory round trips the lambdas add.
Null stays null (addrspacecast preserves it).  Tried and rejected: a plain struct copy (MemCpyOpt
forwards it back to the byref argument -- no change), a word copy (inttoptr -> no provenance -> even
dense s4a goes flat) and an unstamped field-wise copy (also all flat).
Usage: python3 p35_param_copy.py <tree root>"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) / "csrc" / "g2"
FIELDS = {
    "WuParams": "k v q beta A g Wn Wb Un KgA Kn Qn kgB vB H NT ctok clen stop".split(),
    "WyS4AParams": "kgb v_vb A h dv beta v_row dv2 dwT dA db_part T H NT ctok clen dv2c".split(),
    "DavParams": "do_ Vn A dA dVn dOT aNdo scale H NT ctok clen".split(),
}
SCALARS = {"H", "NT", "T", "stop", "scale"}
HELPER = """
// kra P-35: stamp a pointer global (generic -> addrspace(1) -> generic); InferAddressSpaces then
// emits global_* for every access through it, even after the value went through memory (lambdas).
template <typename T>
__device__ __forceinline__ T* g2_glb(T* x) {
    return (T*)(__attribute__((address_space(1))) T*)x;
}
"""


def copy_block(T):
    inits = ", ".join(f"p_.{f}" if f in SCALARS else f"g2_glb(p_.{f})" for f in FIELDS[T])
    return (f"    // kra P-35: field-wise copy, pointers stamped global (global_* ops, not flat)\n"
            f"    const {T} p{{{inits}}};\n")


SITES = [
    ("g2_wu.cuh", "__global__ void __launch_bounds__(256) g2_wu_kernel(WuParams p) {\n",
     "__global__ void __launch_bounds__(256) g2_wu_kernel(WuParams p_) {\n" + copy_block("WuParams")),
    ("g2_wy.cuh", "p3_wy_s4a_kernel(WyS4AParams p) {\n",
     "p3_wy_s4a_kernel(WyS4AParams p_) {\n" + copy_block("WyS4AParams")),
    ("g2_dav.cuh", "__global__ void __launch_bounds__(256) g2_dav_kernel(DavParams p) {\n",
     "__global__ void __launch_bounds__(256) g2_dav_kernel(DavParams p_) {\n" + copy_block("DavParams")),
]
c = (ROOT / "g2_common.h").read_text()
anchor = "__device__ __forceinline__ size_t g2_nat_blk(int i, int hh, int H, int NT,"
assert c.count(anchor) == 1 and "g2_glb" not in c
(ROOT / "g2_common.h").write_text(c.replace(anchor, HELPER + "\n" + anchor))
for f, old, new in SITES:
    pth = ROOT / f
    t = pth.read_text()
    assert t.count(old) == 1, (f, old)
    pth.write_text(t.replace(old, new))
print("P-35 APPLIED")
