"""Round-9 ablation of p3_wy_s4b_kernel's narrow VMEM traffic on the CURRENT baseline (R19); timing
only -- outputs are wrong on purpose.        python3 s4b_ablate.py <tree root> <variant>
  noGather : the prologue's transposed scalar gathers akB/aqB (dA[j][row], dAqk[j][row]) -> 0
  noDwT    : the per-kt dwT B-fragment loads (4 x 8 B per lane, identical across the 4 waves) -> an
             LDS read of the same shape (keeps the MMAC, removes the VMEM)
  noRowKQ  : the per-kt row-major k/q fragment loads (2 x 8 B per lane) -> constants"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wy.cuh"
var = sys.argv[2]
t = P.read_text()


def sub(old, new):
    global t
    assert t.count(old) == 1, (var, old[:80], t.count(old))
    t = t.replace(old, new)


if var == "noGather":
    sub("""            akB[b - A][x] = keep ? dAg[j * RA + row] : 0.f;   // dAT[row][j] == dA[j][row]
            aqB[b - A][x] = (keep && g2_row_ok<M>(j, cl)) ? aq_[j * RA + row] : 0.f;""",
        """            akB[b - A][x] = keep ? 0.001f * (float)(j + row) : 0.f;   // ABLATION
            aqB[b - A][x] = (keep && g2_row_ok<M>(j, cl)) ? 0.002f * (float)(j - row) : 0.f;""")
elif var == "noDwT":
    sub("""            acc = mmac(aa[st], ld8(dwTg + (kc + pm) * kBT + 16 * st + 4 * q), acc);""",
        """            acc = mmac(aa[st], ld8(&sm.stg[buf ^ 1][0][0] + (pm * 64 + 16 * st + 4 * q) % 2048), acc);   // ABLATION""")
elif var == "noRowKQ":
    sub("""        const v4bh kk = rok ? ld8(k_ + ro + kc) : v4bh{0, 0, 0, 0};
        const v4bh qq = rok ? ld8(q_ + ro + kc) : v4bh{0, 0, 0, 0};""",
        """        const v4bh kk = v4bh{(short)(0x3c00 + kc), (short)(0x3c01 + ro), 0x3c02, 0x3c03};   // ABLATION
        const v4bh qq = v4bh{0x3c04, (short)(0x3c05 + kc), 0x3c06, (short)(0x3c07 + ro)};""")
else:
    raise SystemExit(f"unknown variant {var}")
P.write_text(t)
print("ABLATE", var)
