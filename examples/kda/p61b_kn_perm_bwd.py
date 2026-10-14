"""kra P-61b probe (bitwise, on top of P-61): the backward recompute (modes 3/4) also stores h with ds_write_b64.
wu emits Kn with the 16 lanes of every sub-block pi16-permuted (store-side index only), and the recompute reads
the per-k decays in natural k order: k3_g2 passes Gn / Gn_raw re-indexed to natural order (a pure permutation of
the same fp32 values; dhu keeps the permuted Gn).  Probe scope: v8o4w is not taken.
    python3 p61b_kn_perm_bwd.py <tree>"""
import sys
from pathlib import Path

root = Path(sys.argv[1])


def sub(s, old, new, cnt=1):
    assert s.count(old) == cnt, (old[:80], s.count(old))
    return s.replace(old, new)


w = root / "csrc" / "g2" / "g2_wu.cuh"
s = w.read_text()
s = sub(s, "template <bool VL>\n__global__ void __launch_bounds__(256) g2_wu_kernel(WuParams p_) {",
        """// kra P-61b: as wu_emit16, lane (q, m) of every sub-block takes source lane (q, pi16(m))
__device__ __forceinline__ void wu_emit16p(short* dst, const short* tile, int tid) {
    for (int t = tid; t < 1024; t += 256) {
        const int ls = (t & 48) | pi16(t & 15);
        const int o0 = 512 * (t >> 6) + 4 * ls;
        st16x2(dst + 8 * (size_t)t, *reinterpret_cast<const v4bh*>(tile + wu_tile_idx(o0)),
               *reinterpret_cast<const v4bh*>(tile + wu_tile_idx(o0 + 256)));
    }
}

template <bool VL>
__global__ void __launch_bounds__(256) g2_wu_kernel(WuParams p_) {""")
s = sub(s, "                wu_emit16(p.Kn + nblk, tile, tid);", "                wu_emit16p(p.Kn + nblk, tile, tid);   // kra P-61b")
w.write_text(s)

f = root / "csrc" / "g2" / "g2_fwd_v8.cuh"
s = f.read_text()
for o, n in [
    ("case 3: launch4(g2_fwd_h_v8o4_kernel<VL, 3>); break;",
     "case 3: launch4(g2_fwd_h_v8o4_kernel<VL, 3, true, false, false, true>); break;"),
    ("case 4: if (raw4) launch4raw(g2_fwd_h_v8o4_kernel<VL, 4, false>);",
     "case 4: if (raw4) launch4raw(g2_fwd_h_v8o4_kernel<VL, 4, false, false, false, true>);"),
    ("                    else if (sg) launch4(g2_fwd_h_v8o4_kernel<VL, 4, true, true>);",
     "                    else if (sg) launch4(g2_fwd_h_v8o4_kernel<VL, 4, true, true, false, true>);"),
    ("                    else      launch4(g2_fwd_h_v8o4_kernel<VL, 4>);",
     "                    else      launch4(g2_fwd_h_v8o4_kernel<VL, 4, true, false, false, true>);"),
    ("    if (r1a && mode >= 3 && H >= g2_env_long(\"KDA_G2_V8O4W_MIN_H\", 40)",
     "    if (false && r1a && H >= g2_env_long(\"KDA_G2_V8O4W_MIN_H\", 40)"),   # kra P-61b probe scope
]:
    s = sub(s, o, n)
f.write_text(s)

k = root / "hip_kda" / "ops" / "k3_g2.py"
s = k.read_text()
s = sub(s, """            bN_h, Vn, aN_vn = g2_ops.fwd_h_bn(Wn, Kn, Un, Gn, seg=ctx.seg,
                                              emit_an=not no_an, gn_raw=Gn_raw)""",
        """            # kra P-61b: the recompute reads natural-k decays ([a][c][d] = k 16a+4d+c -> k order)
            _nat = lambda t: t.view(-1, 8, 4, 4).transpose(-1, -2).contiguous().view(t.shape)
            bN_h, Vn, aN_vn = g2_ops.fwd_h_bn(Wn, Kn, Un, _nat(Gn), seg=ctx.seg,
                                              emit_an=not no_an, gn_raw=_nat(Gn_raw))""")
k.write_text(s)
print("P-61b APPLIED")
