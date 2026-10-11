"""kra P-26 (round 8, numerics-relaxed by user decision, validated against the FP64 reference_kda.py):
the safe-gate sigmoid lb * 1/(1 + expf(-A*x)) -- accurate expf (range reduction + ldexp + overflow
selects) plus an IEEE division (div_scale/fmas/fixup), ~20 VALU per element -- becomes
rcp(1 + exp2(-A*x*log2e)) with the bare v_exp_f32 / v_rcp_f32 (~4 VALU).  Error: |A*x*log2e| <= ~100
under the K3 gate -> relative error of the sigmoid <~ 1e-5 (fp32), far below bf16 (3.9e-3).
Applied CONSISTENTLY wherever the gate sigmoid is evaluated (forward fwd_prep_a GATE=1, backward
recompute head_gate, and the backward tail's d(sigmoid) recompute), so forward and backward see the
same gate values.  The beta sigmoid is untouched.        Usage: python3 p26_fast_sigmoid.py <tree root>"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) / "csrc" / "g2"


def sub(f, old, new, count=1):
    p = ROOT / f
    t = p.read_text()
    assert t.count(old) == count, (f, old[:70], t.count(old))
    p.write_text(t.replace(old, new))


helper = """
// kra P-26: fast safe-gate sigmoid -- bare v_exp_f32 + v_rcp_f32 (~1e-5 relative in fp32), used
// consistently by every kernel that evaluates the gate sigmoid (fwd_prep_a, head_gate, bwd_tail).
__device__ __forceinline__ float g2_sigmoid_fast(float x) {
    return __builtin_amdgcn_rcpf(1.f + __builtin_amdgcn_exp2f(-x * 1.4426950408889634f));
}
"""
anchor = "__device__ __forceinline__ size_t g2_nat_blk(int i, int hh, int H, int NT,"
sub("g2_common.h", anchor, helper + "\n" + anchor)
for f in ("g2_head.cuh", "g2_fwd_prep.cuh"):
    sub(f, "gt.x = p.lower_bound * (1.f / (1.f + expf(-(A * gt.x))));",
        "gt.x = p.lower_bound * g2_sigmoid_fast(A * gt.x);")
    sub(f, "gt.y = p.lower_bound * (1.f / (1.f + expf(-(A * gt.y))));",
        "gt.y = p.lower_bound * g2_sigmoid_fast(A * gt.y);")
sub("g2_bwd_tail.cuh", "const float sg = 1.f / (1.f + expf(-(A * gg)));",
    "const float sg = g2_sigmoid_fast(A * gg);")
print("P-26 APPLIED")
