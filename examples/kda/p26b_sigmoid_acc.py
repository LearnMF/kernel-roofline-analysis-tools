"""kra P-26b (round 10; numerics-relaxed, FP64-oracle gated with the <=1% rule): P-26's fast gate
sigmoid failed the strict rule because its error was SYSTEMATIC -- the fp32 rounding of -x*log2e
fed a bare v_exp_f32, and dA_log / ddt_bias (sums over T*K) accumulated the bias.  P-26b removes
that bias source and keeps most of the speed:
  y = -x*log2e as yh + yl  (yl = fma(-x, L_hi, -yh) - x*L_lo: the exact rounding error of the
                            product plus the constant's own tail)
  e = exp2(yh) * (1 + yl*ln2)        bare v_exp_f32 on yh; 2^yl to first order (|yl| ~ 2^-24 |yh|)
  r = rcp(1 + e), one Newton step    r += r*(1 - d*r)  (rcp error ~1 ulp -> ~0.5 ulp)
  yh > 127 -> 0                      (d = inf would make the Newton step NaN; IEEE 1/(1+inf) = 0)
~13 VALU vs ~20 for expf + IEEE division.  Same sites as P-26 (fwd_prep_a GATE=1, head_gate,
bwd_tail), applied consistently.          Usage: python3 p26b_sigmoid_acc.py <tree root>"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) / "csrc" / "g2"


def sub(f, old, new, count=1):
    p = ROOT / f
    t = p.read_text()
    assert t.count(old) == count, (f, old[:70], t.count(old))
    p.write_text(t.replace(old, new))


helper = """
// kra P-26b: gate sigmoid 1/(1+exp(-x)) with an FMA-split exp argument (no systematic rounding
// bias) on the bare v_exp_f32 and one Newton step on v_rcp_f32; used by every kernel that
// evaluates the gate sigmoid (fwd_prep_a, head_gate, bwd_tail) so all of them see the same g.
__device__ __forceinline__ float g2_sigmoid_acc(float x) {
    constexpr float kL2eHi = 1.44269502162933349609375f;      // fp32(log2 e)
    constexpr float kL2eLo = 1.925963033500011e-08f;          // log2 e - kL2eHi
    constexpr float kLn2 = 0.693147180559945309f;
    const float yh = -x * kL2eHi;
    const float yl = __builtin_fmaf(-x, kL2eHi, -yh) + (-x) * kL2eLo;
    const float e = __builtin_amdgcn_exp2f(yh) * __builtin_fmaf(yl, kLn2, 1.f);
    const float d = 1.f + e;
    float r = __builtin_amdgcn_rcpf(d);
    r = __builtin_fmaf(__builtin_fmaf(-d, r, 1.f), r, r);
    return yh > 127.f ? 0.f : r;
}
"""
anchor = "__device__ __forceinline__ size_t g2_nat_blk(int i, int hh, int H, int NT,"
sub("g2_common.h", anchor, helper + "\n" + anchor)
for f in ("g2_head.cuh", "g2_fwd_prep.cuh"):
    sub(f, "gt.x = p.lower_bound * (1.f / (1.f + expf(-(A * gt.x))));",
        "gt.x = p.lower_bound * g2_sigmoid_acc(A * gt.x);")
    sub(f, "gt.y = p.lower_bound * (1.f / (1.f + expf(-(A * gt.y))));",
        "gt.y = p.lower_bound * g2_sigmoid_acc(A * gt.y);")
sub("g2_bwd_tail.cuh", "const float sg = 1.f / (1.f + expf(-(A * gg)));",
    "const float sg = g2_sigmoid_acc(A * gg);")
print("P-26b APPLIED")
