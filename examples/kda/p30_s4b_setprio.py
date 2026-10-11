"""kra P-30 (round 9 probe; DCU textbook 6.5.4 BMZ FA): raise the wave's issue priority
(s_setprio 1, fenced by sched_barrier) across s4b_tail's two MMAC passes of each kt (pass K: 8 MMACs
on dh, pass Q: 8 MMACs on h) and drop it (s_setprio 0) before the table/LDS phase, so a wave that
reached its MMAC chain keeps the matrix pipe fed while the other waves issue their loads.
No value changes -> bit-identical.        Usage: python3 p30_s4b_setprio.py <tree root>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wy.cuh"
full = P.read_text()
i0 = full.index("__device__ __forceinline__ void s4b_tail(")   # patch s4b_tail only
i1 = full.index("\n}\n", i0)
t = full[i0:i1]


def sub(old, new):
    global t
    assert t.count(old) == 1, (old[:80], t.count(old))
    t = t.replace(old, new)


sub("""        f4 dkp, dgK;
        {   // pass K tile kt  [verbatim; cs partials replace the fused kernel's atomics]""",
    """        f4 dkp, dgK;
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_setprio(1);     // kra P-30: MMAC passes first
        __builtin_amdgcn_sched_barrier(0);
        {   // pass K tile kt  [verbatim; cs partials replace the fused kernel's atomics]""")
sub("""        f4 ep, en;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float d = gv[e] - __shfl(gv[e], 16 * q + 8, 64);""",
    """        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_setprio(0);     // kra P-30
        __builtin_amdgcn_sched_barrier(0);
        f4 ep, en;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float d = gv[e] - __shfl(gv[e], 16 * q + 8, 64);""")
P.write_text(full[:i0] + t + full[i1:])
print("P-30 APPLIED")
