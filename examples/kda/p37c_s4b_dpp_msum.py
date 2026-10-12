"""kra P-37c (round 12, s4b rewrite step 2a, A-class = bit-identical): the pipelined s4b's msum
(16-lane butterfly x += shfl_xor(x, 1/2/4/8)) through DPP instead of ds_bpermute.

P-37's loop issues 22 ds_bpermute per kt per wave (16 of them msum), each a round trip through
the LDS crossbar with its own lgkmcnt wait -- exposed at one wave per SIMD.  gfx9 DPP has no
xor-4/xor-8 lane pattern, but the butterfly can use row_half_mirror (i <-> 7-i) and row_mirror
(i <-> 15-i): after the xor-1/xor-2 steps all four lanes of a quad hold the same value, after the
third step all eight lanes of a half-row do, so the mirrored partner carries exactly the value the
xor partner would (and IEEE addition is commutative) -> every lane computes the same bits.
Applies to the pipe kernel region only (P-37, optionally P-37b).   python3 p37c_s4b_dpp_msum.py <tree>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wy.cuh"
t = P.read_text()
a = t.index("// ================================================================ kra P-37: pipelined s4b")
b = t.index("p3_wy_s4b_pipe_kernel(WyS4BParams p) {", a)
seg = t[a:b]
n = seg.count("msum(")
assert n >= 1, n
seg = seg.replace("msum(", "s4bp_msum(")
helper = '''
// kra P-37c: DPP butterfly over the 16 lanes of a row, bit-identical to ipi::msum (see p37c notes)
template <int CTRL>
__device__ __forceinline__ float s4bp_dpp(float x) {
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, x), CTRL, 0xF, 0xF, false));
}
__device__ __forceinline__ float s4bp_msum(float x) {
    x += s4bp_dpp<0xB1>(x);     // quad_perm [1,0,3,2]  == xor 1
    x += s4bp_dpp<0x4E>(x);     // quad_perm [2,3,0,1]  == xor 2
    x += s4bp_dpp<0x141>(x);    // row_half_mirror      (partner value == xor 4's)
    x += s4bp_dpp<0x140>(x);    // row_mirror           (partner value == xor 8's)
    return x;
}
'''
anchor = "struct S4BPStage {"
assert seg.count(anchor) == 1
seg = seg.replace(anchor, helper + "\n" + anchor)
# ep/en partner: lane 16q+8 holds g[row 16A+8][kc+4q..] -- the same floats are in the staged g tile,
# so one ds_read_b128 replaces four ds_bpermute (bit-identical operand).
old_ld = "        const f4 gv = *reinterpret_cast<const f4*>(&S.g[row * 16 + 4 * q]);\n"
new_ld = old_ld + "        const f4 g8 = *reinterpret_cast<const f4*>(&S.g[(16 * A + 8) * 16 + 4 * q]);   // kra P-37c\n"
old_ep = "            const float d = gv[e] - __shfl(gv[e], 16 * q + 8, 64);"
new_ep = "            const float d = gv[e] - g8[e];   // == __shfl(gv[e], 16q+8): row 16A+8 of the staged g"
if seg.count(old_ld) == 1 and seg.count(old_ep) == 1:
    seg = seg.replace(old_ld, new_ld).replace(old_ep, new_ep)
    print("ep/en partner from the staged g tile")
else:
    print("ep/en rewrite skipped (pattern not found: P-37b layout?)", seg.count(old_ld), seg.count(old_ep))
t = t[:a] + seg + t[b:]
P.write_text(t)
print(f"P-37c applied: {n} msum call site(s) -> DPP")
