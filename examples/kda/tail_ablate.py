"""Round-10 ablation of g2_bwd_tail_kernel (R19 upper bounds; timing only, outputs wrong on purpose).
    python3 tail_ablate.py <tree root> <variant>
  noShfl   : the l2norm-backward row dot product's 6-step shfl_xor tree (the candidate for the
             textbook's MMAC-as-reduction trick) -> the per-lane partial only
  noSerial : the single-thread fixed-order 512-term dA_part sum -> one term"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_bwd_tail.cuh"
var = sys.argv[2]
t = P.read_text()


def sub(old, new):
    global t
    assert t.count(old) == 1, (var, old[:80], t.count(old))
    t = t.replace(old, new)


if var == "noShfl":
    sub("""                        for (int off = 1; off < 64; off <<= 1)
                            dot += __shfl_xor(dot, off, 64);    // fixed-order tree""",
        """                        for (int off = 1; off < 64 && p.NT < 0; off <<= 1)   // ABLATION
                            dot += __shfl_xor(dot, off, 64);""")
elif var == "noSerial":
    sub("""            for (int k = 0; k < 8 * 64; ++k) s += sDA[k];""",
        """            for (int k = 0; k < 1; ++k) s += sDA[k];   // ABLATION""")
else:
    raise SystemExit(f"unknown variant {var}")
P.write_text(t)
print("ABLATE", var)
