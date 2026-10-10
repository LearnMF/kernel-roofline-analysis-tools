"""Project C probe 2 (compile-only): free s4b's 32 persistent VGPRs of do / v_new fragments
(ado[8], avn[8]) by re-loading them from global (L2-resident) inside the kt loop, then hold
the row-complete dq (and optionally dk) as in probe 1.  Question: does the row-complete s4b
fit in 256 VGPRs without scratch once those fragments stop being persistent?
Usage: python3 c_probe_s4b_reload.py <g2 dir>   (run AFTER c_probe_s4b_rowc.py)"""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "g2_wy.cuh"
t = p.read_text()
a = t.index("template <int A, int M, bool L2R>\n__device__ __forceinline__ void s4b_tail(")
b = t.index("\n}\n", a)
body = t[a:b]
old = """    v4bh ado[8], avn[8];
    {
        const bf16* doN = p.do_ + g2_nat_blk(i, hh, H, 0, true, 4 * 8 * 256);
        #pragma unroll
        for (int vt = 0; vt < 8; ++vt) ado[vt] = ld8(doN + ((A * 8 + vt) * 64 + l) * 4);
        const bf16* vnN = p.v_new + g2_nat_blk(i, hh, H, 0, true, 4 * 8 * 256);
        #pragma unroll
        for (int vt = 0; vt < 8; ++vt) avn[vt] = ld8(vnN + ((A * 8 + vt) * 64 + l) * 4);
    }"""
assert body.count(old) == 1
body = body.replace(old, """    const bf16* doN = p.do_ + g2_nat_blk(i, hh, H, 0, true, 4 * 8 * 256);     // C probe 2: no
    const bf16* vnN = p.v_new + g2_nat_blk(i, hh, H, 0, true, 4 * 8 * 256);   // persistent frags""")
# re-load right before each use (pass K uses avn, pass Q uses ado)
old_k = "            for (int vt = 0; vt < 8; ++vt) a4[vt & 3] = mmac(avn[vt], DHB(vt), a4[vt & 3]);"
old_q = "            for (int vt = 0; vt < 8; ++vt) a4[vt & 3] = mmac(ado[vt], HB(vt), a4[vt & 3]);"
assert body.count(old_k) == 1 and body.count(old_q) == 1
body = body.replace(old_k, "            for (int vt = 0; vt < 8; ++vt) a4[vt & 3] = mmac(ld8(vnN + ((A * 8 + vt) * 64 + l) * 4), DHB(vt), a4[vt & 3]);")
body = body.replace(old_q, "            for (int vt = 0; vt < 8; ++vt) a4[vt & 3] = mmac(ld8(doN + ((A * 8 + vt) * 64 + l) * 4), HB(vt), a4[vt & 3]);")
t = t[:a] + body + t[b:]
p.write_text(t)
print("C PROBE 2 APPLIED")
