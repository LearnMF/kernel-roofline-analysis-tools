"""kra P-37f (round 12, s4b rewrite step 2d, A-class): the prologue's 8 v_new fragment loads issued
back to back + a branch-free C->A transpose.

SQTT/ISA of P-37e: g2_c2a's `if (b1) ... else ...` / `if (b0) {...} else {...}` compile to divergent
branches (s_and_saveexec + s_cbranch_execz), which split each of the 8 avn[vt] = g2_c2a(ld8(...))
into separate basic blocks -> load, s_waitcnt vmcnt(0), transpose, next load: eight serialized
memory round trips per CTA (~10% of wave-cycles at one CTA per CU).  All 8 loads now come first;
the transpose uses selects with the same bit operations (same bits).
Apply after p37 (+c/d/e).          python3 p37f_s4b_c2a_nobranch.py <tree>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wy.cuh"
t = P.read_text()
a = t.index("// ================================================================ kra P-37: pipelined s4b")
b = t.index("p3_wy_s4b_pipe_kernel(WyS4BParams p) {", a)
seg = t[a:b]
old = """            const bf16* vnc = p.vnc + g2_nat_blk(i, hh, H, p.NT, false, 8192) + A * 8 * 256 + l * 4;
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) avn[vt] = g2_c2a(ld8(vnc + vt * 256), l);"""
new = """            const bf16* vnc = p.vnc + g2_nat_blk(i, hh, H, p.NT, false, 8192) + A * 8 * 256 + l * 4;
            v4bh raw_[8];                                   // kra P-37f: all 8 loads in flight first
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) raw_[vt] = ld8(vnc + vt * 256);
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) avn[vt] = s4bp_c2a(raw_[vt], l);"""
assert seg.count(old) == 1, seg.count(old)
seg = seg.replace(old, new)
helper = '''
// kra P-37f: g2_c2a with selects instead of divergent branches (identical bit operations)
__device__ __forceinline__ v4bh s4bp_c2a(v4bh c, int l) {
    uint32_t d0 = (uint32_t)(unsigned short)c[0] | ((uint32_t)(unsigned short)c[1] << 16);
    uint32_t d1 = (uint32_t)(unsigned short)c[2] | ((uint32_t)(unsigned short)c[3] << 16);
    const bool b1 = (l >> 5) & 1, b0 = (l >> 4) & 1;
    uint32_t r = (uint32_t)__shfl_xor((int)(b1 ? d0 : d1), 32, 64);
    d0 = b1 ? r : d0;
    d1 = b1 ? d1 : r;
    const uint32_t snd = b0 ? ((d0 & 0xFFFFu) | (d1 << 16)) : ((d0 >> 16) | (d1 & 0xFFFF0000u));
    r = (uint32_t)__shfl_xor((int)snd, 16, 64);
    const uint32_t e0 = b0 ? ((r & 0xFFFFu) | (d0 & 0xFFFF0000u)) : ((d0 & 0xFFFFu) | (r << 16));
    const uint32_t e1 = b0 ? ((r >> 16) | (d1 & 0xFFFF0000u)) : ((d1 & 0xFFFFu) | (r & 0xFFFF0000u));
    return v4bh{(short)(e0 & 0xFFFFu), (short)(e0 >> 16), (short)(e1 & 0xFFFFu), (short)(e1 >> 16)};
}
'''
anchor = "struct S4BPStage {"
assert seg.count(anchor) == 1
seg = seg.replace(anchor, helper + "\n" + anchor)
t = t[:a] + seg + t[b:]
P.write_text(t)
print("P-37f (branch-free c2a, loads first) APPLIED")
