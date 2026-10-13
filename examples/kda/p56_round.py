"""kra P-56 (B-class candidate): bf16 rounding of the hand-written conversion helper bf_rne / bfr (g2_common.h).
  production: RNE      u += 0x7fff + ((u >> 16) & 1)     (v_bfe_u32 + v_add3_u32 per element)
  rhu       : nearest, ties away from zero   u += 0x8000  (one v_add_u32); differs from RNE only when the
              low 16 bits are exactly 0x8000
  rz        : truncation (no rounding op); biased toward zero by half an ulp on average
__float2bfloat16-based helpers are left untouched.
    python3 p56_round.py <tree> rhu|rz"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_common.h"
mode = sys.argv[2]
t = P.read_text()
old = """__device__ __forceinline__ short bf_rne(float x) {
    uint32_t u = __float_as_uint(x);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (short)(u >> 16);
}"""
new = {"rhu": """__device__ __forceinline__ short bf_rne(float x) {
    uint32_t u = __float_as_uint(x);
    u += 0x8000u;                       // kra P-56 (B-class): nearest, ties away from zero
    return (short)(u >> 16);
}""", "rz": """__device__ __forceinline__ short bf_rne(float x) {
    uint32_t u = __float_as_uint(x);    // kra P-56 (B-class): truncation
    return (short)(u >> 16);
}"""}[mode]
assert t.count(old) == 1
P.write_text(t.replace(old, new))
print("P-56", mode, "APPLIED")
