"""LDS bank-conflict model (gfx9-class LDS: 32 banks x 4 B).

An access pattern is a function lane -> list of byte addresses (one per lane per
instruction; vector accesses give the start address and a width).  Per instruction,
the wave is processed in passes of `lanes_per_pass` lanes; a pass costs
max over banks of the number of DISTINCT dwords mapped to that bank (same dword =
broadcast, no conflict).  Ideal cost = dwords per lane per pass.  Conflict cycles =
cost - ideal.  This is a first-order model used to RANK layouts and predict the
direction/size of SQ_LDS_BANK_CONFLICT changes; it is validated against the counter.
"""
from __future__ import annotations

from collections import defaultdict

BANKS, DW = 32, 4


def instr_cost(addrs: list[int], width: int, lanes_per_pass: int = 32) -> tuple[int, int]:
    """addrs: start byte address per lane (len = wave size); width in bytes."""
    ndw = max(1, width // DW)
    cost = ideal = 0
    for p0 in range(0, len(addrs), lanes_per_pass):
        banks: dict[int, set] = defaultdict(set)
        for a in addrs[p0:p0 + lanes_per_pass]:
            for d in range(ndw):
                dword = a // DW + d
                banks[dword % BANKS].add(dword)
        cost += max(len(s) for s in banks.values())
        ideal += ndw
    return cost, ideal


def pattern_cost(gen, n_instr: int, width: int, wave: int = 64) -> dict:
    """gen(i, lane) -> byte address for instruction i; returns summed cost/ideal."""
    c = i_ = 0
    for i in range(n_instr):
        cc, ii = instr_cost([gen(i, l) for l in range(wave)], width)
        c += cc
        i_ += ii
    return {"cost": c, "ideal": i_, "conflict": c - i_, "degree": c / i_ if i_ else 1.0}
