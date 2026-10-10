"""kra L16: paired 16 B native layout for the recurrence operands (Wn Wb Un Kn KgA Qn QgA AqkA).

Within every 512-element group the two 256-element sub-blocks are interleaved per lane, so the
8 B fragments a lane reads from sub-blocks 2j and 2j+1 form one aligned 16 B word:
    old  o = 256*s + 4*l + e   (s in {0,1}, lane l, element e)   ->   new = 8*l + 4*s + e
nat16() is a bijection inside each 1 KB group and commutes with any 512-aligned base.  Every
producer store goes through nat16(); the recurrence consumers (v8o4/v8o/v7, dhu v8b4/v8b/w8b)
read pairs with one 16 B load; the Python layout code (g2_native.to_*, convert/convert2 for the
affected modes) applies the same permutation.  Values never change -> outputs bit-identical.

Usage: python3 layout16_apply.py <tree>            (hip_kda tree: csrc/g2 + hip_kda/ops)
       python3 layout16_apply.py <dir> --kernels-only   (a bare g2 kernel directory, e.g. flash-train)
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
kernels_only = "--kernels-only" in sys.argv
K = root if kernels_only else root / "csrc/g2"


def edit(path, pairs):
    t = path.read_text()
    for old, new, n in pairs:
        c = t.count(old)
        assert c == n, f"{path.name}: expected {n} x, found {c}: {old[:90]!r}"
        t = t.replace(old, new)
    path.write_text(t)


def edit_in(path, fn_sig, pairs):
    """Apply pairs only inside the function whose signature text is fn_sig."""
    t = path.read_text()
    k0 = t.index(fn_sig)
    k1 = t.index("\n}\n", k0)
    body = t[k0:k1]
    for old, new, n in pairs:
        c = body.count(old)
        assert c == n, f"{path.name}/{fn_sig[:40]}: expected {n} x, found {c}: {old[:90]!r}"
        body = body.replace(old, new)
    path.write_text(t[:k0] + body + t[k1:])


# ---------------------------------------------------------------- helpers
edit(K / "g2_common.h", [(
    "__device__ __forceinline__ f4 ld16f(const float* p) { return *reinterpret_cast<const f4*>(p); }",
    """__device__ __forceinline__ f4 ld16f(const float* p) { return *reinterpret_cast<const f4*>(p); }
// kra L16 paired native layout (Wn Wb Un Kn KgA Qn QgA AqkA): inside every 512-element group
// the two 256-element sub-blocks are interleaved per lane (o = 256s + 4l + e -> 8l + 4s + e),
// so a lane's fragments of sub-blocks 2j and 2j+1 are ONE aligned 16 B load (a dwordx2 costs
// ~16 issue cycles per CU, a dwordx4 ~18-23: measured, kra vmem_issue).  Bijective per 1 KB
// group; commutes with any 512-aligned base.  Values unchanged -> bit-identical.
__device__ __forceinline__ size_t nat16(size_t o) {
    return (o & ~(size_t)511) | (((o >> 2) & 63) << 3) | (((o >> 8) & 1) << 2) | (o & 3);
}
typedef short v8bh_l16 __attribute__((ext_vector_type(8)));
// fragments of sub-blocks 2j, 2j+1: p = base(512-aligned) + 512*j + 8*lane
__device__ __forceinline__ void ld16x2(const bf16* p, v4bh& a, v4bh& b) {
    const v8bh_l16 x = *reinterpret_cast<const v8bh_l16*>(p);
    a = __builtin_shufflevector(x, x, 0, 1, 2, 3);
    b = __builtin_shufflevector(x, x, 4, 5, 6, 7);
}""", 1)])

# ---------------------------------------------------------------- producers
QGA = [("*reinterpret_cast<v4bh*>(p.QgA + qb0) = qg[0];", "*reinterpret_cast<v4bh*>(p.QgA + nat16(qb0)) = qg[0];"),
       ("*reinterpret_cast<v4bh*>(p.QgA + qb0 + 64) = qg[1];", "*reinterpret_cast<v4bh*>(p.QgA + nat16(qb0 + 64)) = qg[1];")]
KN_U = ("*reinterpret_cast<v4bh*>(p.Kn + nblk + 4 * (size_t)U) =", "*reinterpret_cast<v4bh*>(p.Kn + nat16(nblk + 4 * (size_t)U)) =")
AQKA = ("*reinterpret_cast<v4bh*>(p.AqkA + nblk64 + (size_t)(cb + 4 * I) * 256 + 4 * l) = b;",
        "*reinterpret_cast<v4bh*>(p.AqkA + nat16(nblk64 + (size_t)(cb + 4 * I) * 256 + 4 * l)) = b;")
WNUN = ("*reinterpret_cast<v4bh*>(out + nblk + (size_t)(vt + 8 * I) * 256 + 4 * l) = pack4(acc[vt]);",
        "*reinterpret_cast<v4bh*>(out + nat16(nblk + (size_t)(vt + 8 * I) * 256 + 4 * l)) = pack4(acc[vt]);")
OUTC = ("*reinterpret_cast<v4bh*>(outC + nblk + (size_t)(vt + 8 * wv) * 256 + l * 4) =",
        "*reinterpret_cast<v4bh*>(outC + nat16(nblk + (size_t)(vt + 8 * wv) * 256 + l * 4)) =")

edit(K / "g2_fwd_prep.cuh", [(o, n, 1) for o, n in QGA] + [(*KN_U, 1), (*AQKA, 1), (*WNUN, 1)])
edit(K / "g2_fwd_intra2.cuh", [(o, n, 2) for o, n in QGA] + [(*KN_U, 2), (*AQKA, 2), (*WNUN, 2)])
edit(K / "g2_fwd_intra.cuh", [
    ("bstore(p.AqkA, nblk64 + (size_t)(cb + 4 * rb) * 256 + 4 * m + q4 + 64 * e, val);",
     "bstore(p.AqkA, nat16(nblk64 + (size_t)(cb + 4 * rb) * 256 + 4 * m + q4 + 64 * e), val);", 1),
    ("*reinterpret_cast<v4bh*>(p.QgA + nblk + (size_t)(4 * (row & 15) + 64 * q\n                + 256 * d1 + 2048 * d0)) = qg;",
     "*reinterpret_cast<v4bh*>(p.QgA + nat16(nblk + (size_t)(4 * (row & 15) + 64 * q\n                + 256 * d1 + 2048 * d0))) = qg;", 1),
    ("for (int j = 0; j < 4; ++j) p.Kn[ib + 4 * j] = kg[j];", "for (int j = 0; j < 4; ++j) p.Kn[nat16(ib + 4 * j)] = kg[j];", 1),
    (*OUTC, 1),
    ("for (int e = 0; e < 4; ++e) outA[ia + 64 * e] = we[e];", "for (int e = 0; e < 4; ++e) outA[nat16(ia + 64 * e)] = we[e];", 1),
])
edit(K / "g2_wu.cuh", [
    ("*reinterpret_cast<v4bh*>(p.KgA + nblk + (size_t)(4 * (j & 15)\n                + 64 * ((vq & 15) >> 2) + 256 * (vq >> 4) + 2048 * (j >> 4))) =",
     "*reinterpret_cast<v4bh*>(p.KgA + nat16(nblk + (size_t)(4 * (j & 15)\n                + 64 * ((vq & 15) >> 2) + 256 * (vq >> 4) + 2048 * (j >> 4)))) =", 1),
    ("for (int dv = 0; dv < 4; ++dv) p.Kn[nblk + ibn + 4 * dv] = gkv[dv];",
     "for (int dv = 0; dv < 4; ++dv) p.Kn[nat16(nblk + ibn + 4 * dv)] = gkv[dv];", 1),
    (*OUTC, 1),
    ("*reinterpret_cast<v4bh*>(outA + nblk + (size_t)(4 * (l & 15) + 64 * (l >> 4)\n                        + 256 * vt + 2048 * wv)) =",
     "*reinterpret_cast<v4bh*>(outA + nat16(nblk + (size_t)(4 * (l & 15) + 64 * (l >> 4)\n                        + 256 * vt + 2048 * wv))) =", 1),
    ("for (int e = 0; e < 4; ++e) outB[nblk + ib + 4 * e] = we[e];",
     "for (int e = 0; e < 4; ++e) outB[nat16(nblk + ib + 4 * e)] = we[e];", 1),
    ("*reinterpret_cast<v4bh*>(p.Qn + nblk + 4 * t) =", "*reinterpret_cast<v4bh*>(p.Qn + nat16(nblk + 4 * t)) =", 1),
])

# ---------------------------------------------------------------- consumers (16 B pairs)
def pair8(name, var, blk):
    """8 fragments ld8(src + kt*256) -> 4 ld16x2 (src base: 8*blk sub-blocks, even)."""
    return (f"""        const bf16* src = p.{name} + ({blk}) * 256 + l * 4;
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt) {var}[kt] = ld8(src + kt * 256);""",
            f"""        const bf16* src = p.{name} + ({blk}) * 256 + l * 8;      // kra L16: paired
        #pragma unroll
        for (int j = 0; j < 4; ++j) ld16x2(src + j * 512, {var}[2 * j], {var}[2 * j + 1]);""")


W8 = pair8("Wn", "wf", "((chunk + i) * 4 + r1) * 8")
QG8 = pair8("QgA", "qgf", "((chunk + i) * 4 + r1) * 8")
KA8 = pair8("KgA", "ka", "((chunk + i) * 4 + r1) * 8")
AQ4 = ("""        const bf16* src = p.AqkA + (((chunk + i) * 4 + r1) * 4) * 256 + l * 4;
        #pragma unroll
        for (int st = 0; st < 4; ++st) aqf[st] = ld8(src + st * 256);""",
       """        const bf16* src = p.AqkA + (((chunk + i) * 4 + r1) * 4) * 256 + l * 8;   // kra L16: paired
        #pragma unroll
        for (int j = 0; j < 2; ++j) ld16x2(src + j * 512, aqf[2 * j], aqf[2 * j + 1]);""")


def qw(kbexpr):
    return ("""        for (int rt = 0; rt < 4; ++rt) {
            qb[rt] = ld8(p.Qg + (((chunk + i) * 8 + %s) * 4 + rt) * 256 + l * 4);
            wb[rt] = ld8(p.Wb + (((chunk + i) * 8 + %s) * 4 + rt) * 256 + l * 4);
        }""" % (kbexpr, kbexpr),
            """        for (int j = 0; j < 2; ++j) {                    // kra L16: paired
            ld16x2(p.Qg + (((chunk + i) * 8 + %s) * 4) * 256 + j * 512 + l * 8, qb[2 * j], qb[2 * j + 1]);
            ld16x2(p.Wb + (((chunk + i) * 8 + %s) * 4) * 256 + j * 512 + l * 8, wb[2 * j], wb[2 * j + 1]);
        }""" % (kbexpr, kbexpr))


V8 = K / "g2_fwd_v8.cuh"
edit_in(V8, "g2_fwd_h_v8o4_kernel(FwdOParams p)", [
    (*W8, 1), (*QG8, 1), (*AQ4, 1),
    ("""        uf = ld8(p.Un + (((chunk + i) * 4 + r1) * 8 + 2 * vs + v1a_) * 256 + l * 4);""",
     """        // kra L16: uf0/uf1 are one 16 B pair -- loaded on the v1a_ == 0 call (always paired)
        if (v1a_ == 0) ld16x2(p.Un + (((chunk + i) * 4 + r1) * 8 + 2 * vs) * 256 + l * 8, uf0, uf1);
        (void)uf;""", 1),
    ("""        for (int rt = 0; rt < 4; ++rt)
            kf[rt] = ld8(p.Kn + (((chunk + i) * 8 + kb) * 4 + rt) * 256 + l * 4);""",
     """        for (int j = 0; j < 2; ++j)                      // kra L16: paired
            ld16x2(p.Kn + (((chunk + i) * 8 + kb) * 4) * 256 + j * 512 + l * 8, kf[2 * j], kf[2 * j + 1]);""", 1),
])
edit_in(V8, "g2_fwd_h_v8o_kernel(FwdOParams p)", [
    (*W8, 1), (*QG8, 1), (*AQ4, 1),
    ("uf = ld8(p.Un + (((chunk + i) * 4 + r1) * 8 + 2 * vs + v1a) * 256 + l * 4);",
     "uf = ld8(p.Un + nat16((((chunk + i) * 4 + r1) * 8 + 2 * vs + v1a) * 256 + l * 4));", 1),
    ("""        for (int rt = 0; rt < 4; ++rt)
            kf[rt] = ld8(p.Kn + (((chunk + i) * 8 + wv) * 4 + rt) * 256 + l * 4);""",
     """        for (int j = 0; j < 2; ++j)                      // kra L16: paired
            ld16x2(p.Kn + (((chunk + i) * 8 + wv) * 4) * 256 + j * 512 + l * 8, kf[2 * j], kf[2 * j + 1]);""", 1),
])
ENG = K / "g2_engine.cuh"
edit_in(ENG, "g2_fwd_h_v7_kernel(FwdNParams p)", [
    (*W8, 1),
    ("uf[j] = ld8(p.Un + (((chunk + i) * 4 + r1) * 8 + 2 * vs + v1a + j) * 256 + l * 4);",
     "uf[j] = ld8(p.Un + nat16((((chunk + i) * 4 + r1) * 8 + 2 * vs + v1a + j) * 256 + l * 4));", 1),
    ("""            for (int rt = 0; rt < 4; ++rt)
                kf[kl][rt] = ld8(p.Kn + (((chunk + i) * 8 + KPW * wv + kl) * 4 + rt) * 256 + l * 4);""",
     """            for (int j = 0; j < 2; ++j)                  // kra L16: paired
                ld16x2(p.Kn + (((chunk + i) * 8 + KPW * wv + kl) * 4) * 256 + j * 512 + l * 8,
                       kf[kl][2 * j], kf[kl][2 * j + 1]);""", 1),
])
for sig in ("g2_dhu_w8b_kernel(BwdNParams p)", "g2_dhu_v8b_kernel(BwdNParams p)"):
    edit_in(ENG, sig, [(*KA8, 1), (*qw("wv"), 1)])
edit_in(ENG, "g2_dhu_v8b4_kernel(BwdNParams p)", [(*KA8, 1), (*qw("kb"), 1)])

# ---------------------------------------------------------------- python layout code
if not kernels_only:
    NAT = root / "hip_kda/ops/g2_native.py"
    t = NAT.read_text()
    assert "def pair16" not in t
    t = t.replace('''def _perm(x: torch.Tensor, shape, dims) -> torch.Tensor:
    return x.reshape(shape).permute(*dims).contiguous()''', '''def _perm(x: torch.Tensor, shape, dims) -> torch.Tensor:
    return x.reshape(shape).permute(*dims).contiguous()


def pair16(x: torch.Tensor) -> torch.Tensor:
    """kra L16 paired native layout (csrc/g2/g2_common.h nat16): inside every 512-element group
    the two 256-element sub-blocks are interleaved per lane (256s + 4l + e -> 8l + 4s + e)."""
    return x.reshape(-1, 2, 64, 4).transpose(1, 2).contiguous().reshape(x.shape)''', 1)
    for fn in ("to_wn", "to_kn", "to_un", "to_qga", "to_aqka"):
        k0 = t.index(f"def {fn}(")
        k1 = t.index("    return _perm(", k0)
        k2 = t.index("\n", k1)
        line = t[k1:k2]
        t = t[:k1] + line.replace("return _perm(", "return pair16(_perm(", 1) + ")" + t[k2:]
    NAT.write_text(t)

    OPS = root / "hip_kda/ops/g2_ops.py"
    edit(OPS, [
        ('''    return _ext().convert2(x.contiguous(), (g if g is not None else x).contiguous(),
                           int(mode2), like, int(heads), int(chunks))''',
         '''    out = _ext().convert2(x.contiguous(), (g if g is not None else x).contiguous(),
                          int(mode2), like, int(heads), int(chunks))
    return gn.pair16(out) if mode2 in _L16_MODE2 else out''', 1),
        ('''    return _ext().convert(x.contiguous(), int(mode), like.contiguous(), int(heads), int(chunks))''',
         '''    out = _ext().convert(x.contiguous(), int(mode), like.contiguous(), int(heads), int(chunks))
    return gn.pair16(out) if mode in _L16_MODE else out''', 1),
        ('''def convert2(x, mode2, heads, chunks, g=None, T=0):''',
         '''# kra L16: modes whose native output is read by the recurrence kernels in the paired layout
# (g2_common.h nat16); the converter kernels write the plain layout and the pairing is applied here.
_L16_MODE2 = {Mode2.WN, Mode2.KGA, Mode2.KN, Mode2.QN, Mode2.UN, Mode2.QGA, Mode2.AQKA,
              Mode2.QGB, Mode2.KGA2, Mode2.KGB2}
_L16_MODE = {Mode.ToWn, Mode.ToKgA, Mode.ToKn, Mode.ToQn, Mode.ToUn}


def convert2(x, mode2, heads, chunks, g=None, T=0):''', 1),
    ])
print("L16 APPLIED")
