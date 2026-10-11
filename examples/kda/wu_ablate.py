"""Per-part ablation of g2_wu phase 1 ("fill1") on the CURRENT baseline (rule R19); timing only --
outputs are wrong on purpose.     python3 wu_ablate.py <tree root> <variant>
variants: noKgB (scattered kgB emit), noKgA (scattered KgA emit), noKbT (kbT LDS staging),
          noG (the fp32 g loads -> constants), noK (the k loads -> constants)"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wu.cuh"
var = sys.argv[2]
t = P.read_text()


def sub(old, new):
    global t
    assert t.count(old) == 1, (var, old[:80], t.count(old))
    t = t.replace(old, new)


if var == "noKgB":
    sub("            *reinterpret_cast<v4bh*>(p.kgB + wu_vb_quad(j, vq, nblk3)) =",
        "            if (p.stop == 99) *reinterpret_cast<v4bh*>(p.kgB + wu_vb_quad(j, vq, nblk3)) =")
elif var == "noKgA":
    sub("            *reinterpret_cast<v4bh*>(p.KgA + nat16(nblk + (size_t)(4 * (j & 15)",
        "            if (p.stop == 99) *reinterpret_cast<v4bh*>(p.KgA + nat16(nblk + (size_t)(4 * (j & 15)")
elif var == "noKbT":
    sub("""                slab[(size_t)sr * kPadJ2 + wu_slab_col(sr, j)] =
                    bf_rne(bf2f(kv[dv]) * b * eg[dv]);""",
        """                if (p.stop == 99) slab[(size_t)sr * kPadJ2 + wu_slab_col(sr, j)] =
                    bf_rne(bf2f(kv[dv]) * b * eg[dv]);""")
elif var == "noG":
    i = t.index("// ---- phase 1: kbT")
    j = t.index("const f4 gf = ld16f(p.g + sbase + (size_t)g2_row_g<M>(j, cl) * rstride + vq);", i)
    t = t[:j] + "const f4 gf = f4{-0.01f * j, -0.02f, -0.03f, -0.04f * vq};   // ABLATION" + \
        t[j + len("const f4 gf = ld16f(p.g + sbase + (size_t)g2_row_g<M>(j, cl) * rstride + vq);"):]
elif var == "noK":
    i = t.index("// ---- phase 1: kbT")
    old = """            const v4bh kv = g2_row_ok<M>(j, cl) ? *reinterpret_cast<const v4bh*>(p.k + so)
                                                 : v4bh{0, 0, 0, 0};"""
    j = t.index(old, i)
    t = t[:j] + "            const v4bh kv = v4bh{(short)j, (short)vq, 1, (short)so};   // ABLATION" + t[j + len(old):]
else:
    raise SystemExit(f"unknown variant {var}")
P.write_text(t)
print("ABLATE", var)
