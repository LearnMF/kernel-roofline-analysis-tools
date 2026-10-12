"""kra P-46 (A-class, recurrences): the wave index is made wave-uniform (readfirstlane) in
g2_fwd_h_v8o4_kernel, so every per-step address (W/Qg/Aq/U/K/G tiles, HbN/Vn/Ob stores) is computed on
the SALU and the loads use the SGPR-base + 32-bit VGPR-offset form.  SQTT (16K/TP8): the v8o4 step is
issue-bound (436 instructions per wave per step, 49.5% issue + 29% issue delay at 1 wave per SIMD);
~30 of them are 64-bit VALU address adds.  Values unchanged.
python3 p46_rec_uniform_wave.py <tree>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_fwd_v8.cuh"
t = P.read_text()
old = """    __shared__ __align__(16) bf16 oStage[4][256];
    const int tid = threadIdx.x, wv = tid >> 6, l = tid & 63;   // wv in 0..3"""
new = """    __shared__ __align__(16) bf16 oStage[4][256];
    const int tid = threadIdx.x, l = tid & 63;
    const int wv = __builtin_amdgcn_readfirstlane(tid >> 6);    // wv in 0..3 (kra P-46: wave-uniform)"""
assert t.count(old) == 1, t.count(old)
t = t.replace(old, new)
P.write_text(t)
print("P-46 APPLIED")
