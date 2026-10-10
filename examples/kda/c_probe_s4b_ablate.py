"""Project C probe 3: register attribution of s4b by ablation (compile-only; outputs are
WRONG on purpose).  Each variant replaces one persistent operand set of s4b_tail with a
cheap stand-in and records VGPR / scratch, to find which state drives s4b to the 256-VGPR cap
(the cap blocks the row-complete fusion, the dg-path fusion and a prefetch of the per-kt row
loads alike).  Usage: python3 c_probe_s4b_ablate.py <g2 dir> <variant>"""
import sys
from pathlib import Path

p, var = Path(sys.argv[1]) / "g2_wy.cuh", sys.argv[2]
t = p.read_text()
a = t.index("__device__ __forceinline__ void s4b_tail(")
b = t.index("\n}\n", a)
body = t[a:b]


def rep(old, new):
    global body
    assert body.count(old) >= 1, (var, old[:60])
    body = body.replace(old, new)


if var == "no_ado":       # do fragments (8 x v4bh = 16 VGPR) -> constant
    rep("for (int vt = 0; vt < 8; ++vt) ado[vt] = ld8(doN + ((A * 8 + vt) * 64 + l) * 4);",
        "for (int vt = 0; vt < 8; ++vt) ado[vt] = v4bh{1, 1, 1, 1};")
elif var == "no_avn":     # v_new fragments -> constant
    rep("for (int vt = 0; vt < 8; ++vt) avn[vt] = ld8(vnN + ((A * 8 + vt) * 64 + l) * 4);",
        "for (int vt = 0; vt < 8; ++vt) avn[vt] = v4bh{1, 1, 1, 1};")
elif var == "no_dak":     # dA row block (4 x f4 = 16 VGPR) -> constant
    rep("for (int ut = 0; ut < 4; ++ut) dak[ut] = ld16f(dAg + row * RA + 16 * ut + 4 * q);",
        "for (int ut = 0; ut < 4; ++ut) dak[ut] = f4{1.f, 1.f, 1.f, 1.f};")
elif var == "no_tables":  # aqA / aqB / akB (24-52 VGPR by A) -> constants
    rep("aqA[b] = rok ? ld16f(aq_ + row * RA + 16 * b + 4 * q) : f4{0.f, 0.f, 0.f, 0.f};",
        "aqA[b] = f4{1.f, 1.f, 1.f, 1.f};")
    rep("akB[b - A][x] = keep ? dAg[j * RA + row] : 0.f;   // dAT[row][j] == dA[j][row]",
        "akB[b - A][x] = 1.f;")
    rep("aqB[b - A][x] = (keep && g2_row_ok<M>(j, cl)) ? aq_[j * RA + row] : 0.f;",
        "aqB[b - A][x] = 1.f;")
elif var == "no_prefetch":  # h/dh next-tile prefetch held in registers across the step
    rep("""        if (kt + 2 < 8) {
            pfh = hg4[(kt + 2) * 256 + threadIdx.x];
            pfd = dhg4[(kt + 2) * 256 + threadIdx.x];
        }""", "")
    rep("""            stg4[(buf * 2 + 0) * 256 + threadIdx.x] = pfh;
            stg4[(buf * 2 + 1) * 256 + threadIdx.x] = pfd;""",
        """            stg4[(buf * 2 + 0) * 256 + threadIdx.x] = hg4[(kt + 2) * 256 + threadIdx.x];
            stg4[(buf * 2 + 1) * 256 + threadIdx.x] = dhg4[(kt + 2) * 256 + threadIdx.x];""")
else:
    raise SystemExit(f"unknown variant {var}")
t = t[:a] + body + t[b:]
p.write_text(t)
print("ABLATE", var)
