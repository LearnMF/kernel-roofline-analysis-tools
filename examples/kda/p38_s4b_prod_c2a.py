"""kra P-38 (A-class, production s4b, 2 CTAs/CU): the P-37f prologue fix in the production kernel --
the 8 C-form v_new loads issued back to back and a branch-free C->A transpose (g2_c2a's divergent
if/else split every load into its own basic block: load, vmcnt(0), transpose, next load).
python3 p38_s4b_prod_c2a.py <tree>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wy.cuh"
t = P.read_text()
old = """            const bf16* vnc = p.vnc + g2_nat_blk(i, hh, H, p.NT, false, 8192) + A * 8 * 256 + l * 4;
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) avn[vt] = g2_c2a(ld8(vnc + vt * 256), l);"""
assert t.count(old) == 1, t.count(old)
new = """            const bf16* vnc = p.vnc + g2_nat_blk(i, hh, H, p.NT, false, 8192) + A * 8 * 256 + l * 4;
            v4bh raw_[8];                                   // kra P-38: all 8 loads in flight first
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) raw_[vt] = ld8(vnc + vt * 256);
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) avn[vt] = g2_c2a_sel(raw_[vt], l);"""
t = t.replace(old, new)
helper = '''// kra P-38: g2_c2a with selects instead of divergent branches (identical bit operations)
__device__ __forceinline__ v4bh g2_c2a_sel(v4bh c, int l) {
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
anchor = "struct WyS4AParams {"
assert t.count(anchor) == 1
t = t.replace(anchor, helper + anchor)
P.write_text(t)
print("P-38 APPLIED")
