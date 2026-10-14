"""Global layout assignment model for the KDA interface tensors (phase 2, step 4 preparation).

Problem: every materialized interface tensor has ONE producer and one or more consumer kernels; each side touches it
with a fixed thread mapping (its ROLE).  The layout chosen for the tensor decides each side's access PATTERN (bytes
per lane x rows per wave instruction x contiguous bytes per row) and therefore its TA cost; a mismatching layout can
also be served by an on-chip transpose (LDS exchange).  Choosing all layouts together (optionally adding a copy) to
minimize the summed cost is the global problem; P-17 / P-20 / P-22 were local instances of it.

This module holds
  ROLE x LAYOUT -> pattern rules (derived from the fragment geometry, R47: rows x bytes per instruction),
  the access INVENTORY of the production kernels (per wave, from the sources; s4b and bwd_tail so far),
  a cost hook: blocks_model.vmem_cost when the building-block library exists, else the old TA table
  (results/kda-bound-20261012), which is known to miss several patterns (see `validate`).
    python3 kda_layout.py validate | candidates | patterns
"""
import json
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CARDS = os.path.join(ROOT, "results", "kda-ceiling-cards-20261013", "cards.json")


@dataclass(frozen=True)
class Pattern:
    bpl: int          # bytes per lane
    rows: int         # distinct rows touched by one wave instruction
    row_bytes: int    # contiguous bytes per row
    store: bool = False

    def key(self):
        return f"{'st' if self.store else 'ld'} {self.bpl}B/lane {self.rows}x{self.row_bytes}B"


# ------------------------------------------------------------------------------------------- roles
# A role = how one wave instruction of the kernel touches the tensor:
#   frag8   : MMAC operand fragment, lane (m, q) = row m, 4 consecutive bf16 along the reduction dim (8 B)
#   frag16f : fp32 accumulator-shaped tile, lane (m, q) = row m, 4 consecutive fp32 (16 B)
#   row2f   : lane owns 2 adjacent fp32 of ONE row (the wave covers one 512 B row)      (tail)
#   row2h   : lane owns 2 adjacent bf16 of one row (one 256 B row)                        (tail)
#   tile4x  : 16 B per lane over 4 rows x 256 B (row-blocked tiles)
#   gather4 : 4 B per lane, 4 rows x 64 B (s4b aqB / akB transposed reads, R47)
#   rowvec  : 4 B per lane, one value per row for 16 / 64 rows (rstd, beta, db)
ROLES = {"frag8", "frag16f", "row2f", "row2h", "tile4x", "gather4", "rowvec16", "rowvec64", "native16"}


def pattern(role, layout, store=False):
    """Pattern of one wave instruction for (role, layout).  'native*' layouts are the fragment-contiguous
    natives (one wave instruction = 64 lanes x bpl contiguous); 'row' is row-major [T, H, X]."""
    nat = layout.startswith("nat") or layout in ("bN", "aN", "vB")
    if role == "frag8":
        return Pattern(8, 1, 512, store) if nat else Pattern(8, 16, 32, store)
    if role == "frag16f":
        return Pattern(16, 1, 1024, store) if nat else Pattern(16, 16, 64, store)
    if role == "native16":
        return Pattern(16, 1, 1024, store)
    if role == "row2f":
        return Pattern(8, 1, 512, store)            # a row-wise consumer reads natives contiguously too
    if role == "row2h":
        return Pattern(4, 1, 256, store)
    if role == "tile4x":
        return Pattern(16, 4, 256, store)
    if role == "gather4":
        return Pattern(4, 4, 64, store)
    if role == "rowvec16":
        return Pattern(4, 16, 4, store)
    if role == "rowvec64":
        return Pattern(4, 64, 4, store)
    raise KeyError(role)


# ------------------------------------------------------------------------------------------- inventory
@dataclass(frozen=True)
class Access:
    kernel: str
    tensor: str       # materialization tag (kda_ir names)
    role: str
    n: float          # wave instructions per wave (C = 64, 4 waves per CTA, dense)
    store: bool = False
    layout: str = "row"
    movable: bool = True   # False: the layout is fixed by the model interface (inputs / outputs) or a scalar row
    note: str = ""
    order: bool = False    # the consumer reduces / scans with a fixed lane order (bitwise): a non-row layout needs an
                           # on-chip remap (LDS exchange) back to the production lane mapping


INVENTORY = [
    # ---- wy_s4b (g2_wy.cuh s4b_tail, per wave A; aqA / aqB averaged over A = 0..3)
    Access("wy_s4b", "dA", "frag16f", 4, note="dak: 4 f4 rows of s4a's fp32 dA"),
    Access("wy_s4b", "dAqk", "frag16f", 2.5, note="aqA: A+1 f4 rows"),
    Access("wy_s4b", "dA", "gather4", 10, note="akB: transposed scalar reads"),
    Access("wy_s4b", "dAqk", "gather4", 10, note="aqB: transposed scalar reads"),
    Access("wy_s4b", "do", "frag8", 8, movable=False, note="ado: row-major do (P-20 removed the aN_do copy)"),
    Access("wy_s4b", "Vn", "frag8", 8, layout="natC", note="avn: C-form Vn + in-register transpose (P-22a)"),
    Access("wy_s4b", "Akk", "frag8", 4, note="aa: T^-1 fragments (prologue)"),
    Access("wy_s4b", "h", "native16", 8, layout="bN", note="h tiles, 16 B / thread, staged to LDS"),
    Access("wy_s4b", "dh", "native16", 8, layout="bN", note="dh tiles"),
    Access("wy_s4b", "dwT", "tile4x", 8, layout="natB", note="dwv: one 8 B load per thread per kt (4 rows x 128 B)"),
    Access("wy_s4b", "g", "frag16f", 8, note="gv: fp32 g tile per kt (16 rows x 64 B)"),
    Access("wy_s4b", "g", "rowvec16", 8, note="gn: chunk-end g row per kt (one row, 4 quads)"),
    Access("wy_s4b", "kn", "frag8", 8, note="kk per kt"),
    Access("wy_s4b", "qn", "frag8", 8, note="qq per kt"),
    Access("wy_s4b", "dq", "frag16f", 8, store=True), Access("wy_s4b", "dk", "frag16f", 8, store=True),
    Access("wy_s4b", "dg", "frag16f", 8, store=True),
    # ---- bwd_tail (g2_bwd_tail.cuh; warp w owns rows 16w..16w+15, lane owns 2 adjacent columns)
    Access("bwd_tail", "dt_bias", "row2f", 1, movable=False),
    Access("bwd_tail", "dg", "row2f", 16, order=True, note="phase A: dg rows (reverse cumsum)"),
    Access("bwd_tail", "g_raw", "row2h", 16, movable=False),
    Access("bwd_tail", "dg_raw", "row2h", 16, store=True, movable=False),
    Access("bwd_tail", "rstd", "rowvec16", 2, note="q_rstd / k_rstd once per warp (P-54a)"),
    Access("bwd_tail", "dq", "row2f", 16, order=True), Access("bwd_tail", "dk", "row2f", 16, order=True),
    Access("bwd_tail", "qn", "row2h", 16, order=True), Access("bwd_tail", "kn", "row2h", 16, order=True),
    Access("bwd_tail", "dq_raw", "row2h", 16, store=True, movable=False),
    Access("bwd_tail", "dk_raw", "row2h", 16, store=True, movable=False),
    Access("bwd_tail", "beta/db", "rowvec64", 0.75, note="wave 0 only: beta_raw, db loads, db_out store"),
    Access("bwd_tail", "partials", "rowvec16", 0.5, store=True),
]


# ------------------------------------------------------------------------------------------- costs
# old TA table (CU cycles per wave instruction, data in HBM; results/kda-bound-20261012 README)
OLD_TA = {("ld", 16, 1, 1024): 21, ("ld", 8, 1, 512): 22, ("ld", 4, 1, 256): 10, ("ld", 16, 4, 256): 43,
          ("ld", 16, 16, 64): 169, ("ld", 8, 16, 32): 85, ("ld", 4, 64, 4): 229,
          ("st", 16, 1, 1024): 56, ("st", 8, 1, 512): 28, ("st", 16, 4, 256): 56, ("st", 16, 16, 64): 67,
          ("st", 8, 16, 32): 67}


def cost(p):
    """TA cycles of one wave instruction: the building-block library when present, else the old table."""
    try:
        import blocks_model  # noqa: PLC0415
        return blocks_model.vmem_cost(p.bpl, p.rows, p.row_bytes, "hbm", store=p.store)["ta_cycles"], "blocks"
    except Exception:
        k = ("st" if p.store else "ld", p.bpl, p.rows, p.row_bytes)
        return OLD_TA.get(k), "old"


def kernel_ta(kernel, inv=INVENTORY, override=None):
    """predicted TA cycles and VMEM instructions per wave; override: {tensor: layout}"""
    tot, n, missing = 0.0, 0.0, set()
    for a in inv:
        if a.kernel != kernel:
            continue
        lay = (override or {}).get(a.tensor, a.layout)
        p = pattern(a.role, lay, a.store)
        c, src = cost(p)
        n += a.n
        if c is None:
            missing.add(p.key())
            continue
        tot += c * a.n
    return tot, n, missing


def validate():
    cards = json.load(open(CARDS))
    print("| kernel | shape | VMEM/wave inventory | VMEM/wave PMC | TA/wave predicted (old table) | TA/wave measured | missing pattern costs |")
    print("|---|---|---:|---:|---:|---:|---|")
    for k, card in (("wy_s4b", "wy_s4b#1"), ("bwd_tail", "bwd_tail#1")):
        for sh in ("16384_12", "16384_48"):
            c = cards[sh]["cards"][card]
            waves = c["ctas"] * c["block"] // 64
            ta, n, miss = kernel_ta(k)
            print(f"| {k} | {sh} | {n:.1f} | {c['vmem'] / waves:.1f} | {ta:.0f} | {c['ta_busy'] / waves:.0f} | "
                  f"{'; '.join(sorted(miss)) or '-'} |")


def candidates():
    """per movable tensor of the inventoried kernels: each access's pattern under every candidate layout."""
    lays = ("row", "natA", "natB", "natC")
    tens = sorted({a.tensor for a in INVENTORY if a.movable})
    print("| tensor | access (kernel role n/wave) | " + " | ".join(lays) + " |")
    print("|---|---|" + "---|" * len(lays))
    for t in tens:
        for a in INVENTORY:
            if a.tensor != t or not a.movable:
                continue
            cells = []
            for lay in lays:
                p = pattern(a.role, lay, a.store)
                c, _ = cost(p)
                rm = " +LDS remap" if a.order and lay != "row" else ""
                cells.append(f"{p.key()} = {c if c is not None else '?'}{rm}")
            print(f"| {t} | {a.kernel} {a.role}{' st' if a.store else ''} x{a.n:g} | " + " | ".join(cells) + " |")


def patterns():
    """every pattern the inventory needs, with / without a cost (the list for the building-block library)"""
    need = {}
    for a in INVENTORY:
        for lay in (a.layout, "row", "natA"):
            p = pattern(a.role, lay, a.store)
            need[p.key()] = cost(p)[0]
    for k in sorted(need):
        print(f"{k:32s} {need[k] if need[k] is not None else 'MISSING'}")


if __name__ == "__main__":
    {"validate": validate, "candidates": candidates, "patterns": patterns}[sys.argv[1] if len(sys.argv) > 1 else "validate"]()
