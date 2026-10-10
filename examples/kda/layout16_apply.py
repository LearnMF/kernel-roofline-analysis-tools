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
# Pairing is a property of the TENSOR, not of the layout form: the same A/B/C-form functions
# and converter modes also build dOT, dVn, Vn (read unpaired).  So the form functions stay plain
# and the pairing is applied explicitly where one of the eight paired tensors is built.
if not kernels_only:
    NAT = root / "hip_kda/ops/g2_native.py"
    edit(NAT, [('''def _perm(x: torch.Tensor, shape, dims) -> torch.Tensor:
    return x.reshape(shape).permute(*dims).contiguous()''', '''def _perm(x: torch.Tensor, shape, dims) -> torch.Tensor:
    return x.reshape(shape).permute(*dims).contiguous()


def pair16(x: torch.Tensor) -> torch.Tensor:
    """kra L16 paired native layout (csrc/g2/g2_common.h nat16) of a plain native tensor: inside
    every 512-element group the two 256-element sub-blocks are interleaved per lane
    (256s + 4l + e -> 8l + 4s + e).  Applies to Wn Wb Un Kn KgA Qn QgA AqkA only."""
    return x.reshape(-1, 2, 64, 4).transpose(1, 2).contiguous().reshape(x.shape)


def unpair16(x: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pair16`."""
    return x.reshape(-1, 64, 2, 4).transpose(1, 2).contiguous().reshape(x.shape)''', 1)])

    OPS = root / "hip_kda/ops/g2_ops.py"
    edit(OPS, [
        ('''def convert2(x, mode2, heads, chunks, g=None, T=0):''',
         '''def convert2(x, mode2, heads, chunks, g=None, T=0, pair16=False):''', 1),
        ('''    return _ext().convert2(x.contiguous(), (g if g is not None else x).contiguous(),
                           int(mode2), like, int(heads), int(chunks))''',
         '''    out = _ext().convert2(x.contiguous(), (g if g is not None else x).contiguous(),
                          int(mode2), like, int(heads), int(chunks))
    # kra L16: pair16=True when the output is one of the paired tensors (Wn Wb Un Kn KgA Qn QgA AqkA)
    return gn.pair16(out) if pair16 else out''', 1),
        ('''def convert(x: torch.Tensor, mode: int, like: torch.Tensor, heads: int, chunks: int) -> torch.Tensor:''',
         '''def convert(x: torch.Tensor, mode: int, like: torch.Tensor, heads: int, chunks: int,
            pair16: bool = False) -> torch.Tensor:''', 1),
        ('''    return _ext().convert(x.contiguous(), int(mode), like.contiguous(), int(heads), int(chunks))''',
         '''    out = _ext().convert(x.contiguous(), int(mode), like.contiguous(), int(heads), int(chunks))
    return gn.pair16(out) if pair16 else out''', 1),
        ('''    return (gn.to_wn(w_row, H), gn.to_kn(kg_row, H), gn.to_un(u_row, H), gn.to_gn(g_row, H))''',
         '''    return (gn.pair16(gn.to_wn(w_row, H)), gn.pair16(gn.to_kn(kg_row, H)),
            gn.pair16(gn.to_un(u_row, H)), gn.to_gn(g_row, H))''', 1),
        ('''    return (gn.to_wn(kg_row, H), gn.to_kn(qg_row, H), gn.to_kn(w_row, H),
            gn.to_kn(do_row, H), gn.to_un(dv_row, H), gn.to_gn(g_row, H))''',
         '''    return (gn.pair16(gn.to_wn(kg_row, H)), gn.pair16(gn.to_kn(qg_row, H)), gn.pair16(gn.to_kn(w_row, H)),
            gn.to_kn(do_row, H), gn.to_un(dv_row, H), gn.to_gn(g_row, H))''', 1),
    ])
    edit(root / "hip_kda/ops/k3_g2.py", [(f"{v} = g2_ops.convert2({a}, g2_ops.Mode2.{m}, H, NT{x})",
                                          f"{v} = g2_ops.convert2({a}, g2_ops.Mode2.{m}, H, NT{x}, pair16=True)", 1)
                                         for v, a, m, x in (("Wn", "w", "WN", ""), ("Kn", "kg", "KN", ""),
                                                            ("Un", "u", "UN", ""), ("QgA", "qn", "QGA", ", g=g"),
                                                            ("AqkA", "Aqk", "AQKA", ""))])
    T = root / "tests/g2"
    edit(T / "test_wu_gate.py", [(
        '''    ours = {"Wn": Wn, "Wb": Wb, "Un": Un,''',
        '''    ours = {"Wn": gn.unpair16(Wn), "Wb": gn.unpair16(Wb), "Un": gn.unpair16(Un),   # kra L16''', 1)])
    edit(T / "test_s0_bitexact.py", [
        ('''    return gn.to_wn(w, H), gn.to_kn(kg, H), gn.to_un(u, H), gn.to_gn(gk, H)''',
         '''    return (gn.pair16(gn.to_wn(w, H)), gn.pair16(gn.to_kn(kg, H)), gn.pair16(gn.to_un(u, H)),
            gn.to_gn(gk, H))                                   # kra L16: P1 reads paired''', 1),
        ('''    KgA = gn.to_wn(kg, H)
    Qn = gn.to_kn(qg, H)
    Wb = gn.to_kn(w, H)''', '''    KgA = gn.pair16(gn.to_wn(kg, H))                       # kra L16: P2 reads paired
    Qn = gn.pair16(gn.to_kn(qg, H))
    Wb = gn.pair16(gn.to_kn(w, H))''', 1)])
    edit(T / "test_fi_gate.py", [
        ('''        "Wn": (Wn, cv2(w, M2.WN, H, NT)),
        "Kn": (Kn, cv2(kg, M2.KN, H, NT)),''',
         '''        "Wn": (Wn, cv2(w, M2.WN, H, NT, pair16=True)),        # kra L16
        "Kn": (Kn, cv2(kg, M2.KN, H, NT, pair16=True)),''', 1)])
    t = (T / "test_fi_gate.py").read_text()
    for o, n in (('cv2(u, M2.UN, H, NT)', 'cv2(u, M2.UN, H, NT, pair16=True)'),
                 ('cv2(q, M2.QGA, H, NT, g=g)', 'cv2(q, M2.QGA, H, NT, g=g, pair16=True)')):
        assert t.count(o) == 1, ("fi_gate", o, t.count(o))
        t = t.replace(o, n)
    o = '''                            .to(torch.bfloat16), M2.AQKA, H, NT)),'''
    assert t.count(o) == 1, ("fi_gate AqkA", t.count(o))
    t = t.replace(o, '''                            .to(torch.bfloat16), M2.AQKA, H, NT, pair16=True)),''')
    (T / "test_fi_gate.py").write_text(t)
    edit(T / "test_fi2_gate.py", [(
        '''    fla = {"Wn": cv2(w, M2.WN, H, NT), "Kn": cv2(kg, M2.KN, H, NT), "Un": cv2(u, M2.UN, H, NT),
           "QgA": cv2(q, M2.QGA, H, NT, g=g), "AqkA": cv2(Aqk_m.to(torch.bfloat16), M2.AQKA, H, NT),''',
        '''    fla = {"Wn": cv2(w, M2.WN, H, NT, pair16=True), "Kn": cv2(kg, M2.KN, H, NT, pair16=True),
           "Un": cv2(u, M2.UN, H, NT, pair16=True),           # kra L16: paired natives
           "QgA": cv2(q, M2.QGA, H, NT, g=g, pair16=True),
           "AqkA": cv2(Aqk_m.to(torch.bfloat16), M2.AQKA, H, NT, pair16=True),''', 1)])
print("L16 APPLIED")
