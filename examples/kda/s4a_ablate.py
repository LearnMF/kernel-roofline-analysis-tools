"""Round-9 ablation of p3_wy_s4a_kernel's wave-invariant VMEM traffic on the CURRENT baseline (R19);
timing only -- outputs are wrong on purpose.     python3 s4a_ablate.py <tree root> <variant>
  noKgB   : pass W's kgB B-fragment loads (4 x 8 B per lane per kt, identical across the 4 waves)
            -> an LDS read of the same shape
  noAfrag : the dA stage's A fragment loads (16 x 8 B per lane, identical across waves) -> LDS
  noBeta  : the dA stage's 16 scalar beta loads per lane (identical across waves) -> constants"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wy.cuh"
var = sys.argv[2]
t = P.read_text()


def sub(old, new):
    global t
    assert t.count(old) == 1, (var, old[:80], t.count(old))
    t = t.replace(old, new)


if var == "noKgB":
    sub("""                    const v4bh kgb = ld8(kgB + ((st * 8 + kt) * 64 + l) * 4);""",
        """                    const v4bh kgb = ld8(reinterpret_cast<const bf16*>(XT) + ((st * 64 + l) * 4) % 4096);   // ABLATION""")
elif var == "noAfrag":
    sub("""                xc[ut] = mmac(da[st], aok ? ld8(A_ + (16 * ut + pm) * RA + 16 * st + 4 * q)
                                          : v4bh{0, 0, 0, 0}, xc[ut]);""",
        """                xc[ut] = mmac(da[st], aok ? ld8(reinterpret_cast<const bf16*>(XT) + (16 * ut + pm) * 64 + 16 * st + 4 * q)
                                          : v4bh{0, 0, 0, 0}, xc[ut]);   // ABLATION""")
elif var == "noBeta":
    sub("""                da[st][e] = bfr(tr > s ? dA[st][e] * (g2_row_ok<M>(s, cl) ? beta_[s * H] : 0.f) : 0.f);""",
        """                da[st][e] = bfr(tr > s ? dA[st][e] * (g2_row_ok<M>(s, cl) ? 0.5f + 0.001f * s : 0.f) : 0.f);   // ABLATION""")
else:
    raise SystemExit(f"unknown variant {var}")
P.write_text(t)
print("ABLATE", var)
