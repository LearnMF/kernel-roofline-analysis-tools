"""Per-part ablation of g2_dav (timing only -- outputs are WRONG on purpose): each variant removes
one memory stream so its share of dav's time can be read off probe_dav.py (rule R19: ablations
are re-done on the current baseline).    python3 dav_ablate.py <tree root> <variant>
variants: noANdo (the scattered aN_do emit), noAT (the 16 scalar A^T gathers per lane),
          noDo1 (pass-1 do fragment loads), noVn (the Vn staging loads),
          noConf2 (pass-2 do^T staging writes conflict-free, same instruction count)"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_dav.cuh"
var = sys.argv[2]
t = P.read_text()


def sub(old, new):
    global t
    assert t.count(old) == 1, (var, old[:80])
    t = t.replace(old, new)


if var == "noANdo":
    sub("            *reinterpret_cast<v4bh*>(p.aNdo + nblk3 + (size_t)(4 * (j & 15)",
        "            if (p.scale == 12345.f) *reinterpret_cast<v4bh*>(p.aNdo + nblk3 + (size_t)(4 * (j & 15)")
elif var == "noAT":
    sub("""                    af[e] = (t >= tp && g2_row_ok<M>(t, cl)) ?           // causal keep
                        *reinterpret_cast<const short*>(
                            p.A + g2_rm_tok(t0 + t, hh, p.H, kBT) + tp) : 0;""",
        """                    af[e] = (t >= tp) ? (short)(t + tp) : (short)0;   // ABLATION""")
elif var == "noDo1":
    sub("""                const v4bh df = dok ? *reinterpret_cast<const v4bh*>(
                    p.do_ + g2_rm_tok(t0 + rdo, hh, p.H, kV)
                    + 16 * vs + (l >> 4) * 4) : v4bh{0, 0, 0, 0};""",
        """                const v4bh df = v4bh{(short)vs, (short)rdo, (short)dok, 1};   // ABLATION""")
elif var == "noVn":
    sub("""            const v4bh vq = *reinterpret_cast<const v4bh*>(p.Vn + nblk + (size_t)sb * 256 + c * 4);""",
        """            const v4bh vq = v4bh{(short)sb, (short)c, 1, 2};   // ABLATION""")
elif var == "noConf2":   # pass-2 do^T staging writes: same 32 ds_write_b16 per lane, one dword per lane (no conflicts)
    old_w = """            for (int d = 0; d < 4; ++d)
                slab[(size_t)(vq + d) * kPadJ3 + j] = dv_[d];
            // T3: the SAME do loads also emit P2's dOT (B form, stride-4 quads, std
            // block order) and P3's aN_do (A form, plain row nibble, P3 block order)
            // -- the two B3.5 convert2 calls (which re-read do, ~200 MB @8K/48)
            // disappear.  Index math identical to g2_wu's verified Kn/KgA emits.
        }
        } else {"""
    sub(old_w, old_w.replace("slab[(size_t)(vq + d) * kPadJ3 + j] = dv_[d];",
                             "slab[(size_t)((it_ * 4 + d) & 7) * 1024 + 2 * tid] = dv_[d];   // ABLATION"))
else:
    raise SystemExit(f"unknown variant {var}")
P.write_text(t)
print("ABLATE", var)
