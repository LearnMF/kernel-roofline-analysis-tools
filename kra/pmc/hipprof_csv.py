"""Per-kernel PMC records from DTK hipprof `--pmc-read` / `--pmc-write` CSV (`--pmc-type 3`).

Counter semantics were calibrated on gfx936 with kernels of exactly known work
(`microbench pmccal`, see docs/methodology.md):
  * HBM bytes   read  = 64*(TCC_EA_RDREQ - TCC_EA_RDREQ_32B) + 32*TCC_EA_RDREQ_32B
                write = 64*TCC_EA_WRREQ_64B + 32*(TCC_EA_WRREQ - TCC_EA_WRREQ_64B)
                (+ the TCC_EA1_* instances, zero on gfx936)
  * SQ_INSTS_VALU counts MMAC instructions too (wave-level instructions).
  * SQ_ACTIVE_INST_VALU counts VALU-pipe issue slots: 1 per VALU, 2 per MMAC
    16x16x16, 4 per transcendental.  pipe utilization = ACTIVE / (GRBM_GUI_ACTIVE * CUs)
    reaches 0.97-1.00 at the measured MMAC / packed-FMA / exp2 peaks.
  * ACTIVE - INSTS = extra slots of multi-slot instructions = MMAC count when the
    kernel has no transcendentals (an upper bound otherwise).
  * TA_TA_BUSY: 16 instances (one texture-addresser per 5 CUs on the 80-CU part); the SUM
    over instances / (GRBM_GUI_ACTIVE * CUs) is the vector-memory (TA) issue utilization:
    0.91-0.93 on the saturated `microbench vmem_issue`, equal to hipprof's derived "L1 cache
    unit is active" (calibrated 2026-10-10).  This is the pipe the KDA recurrences were bound
    by (validation P-7/L16), invisible to the HBM and VALU SOLs.
The read and write presets come from two separate runs; rows are joined by order.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

INST = re.compile(r"\[(\d+)\]$")
META = ("KernelName", "grd", "wgr", "lds", "scr", "arch_vgpr", "accum_vgpr", "sgpr",
        "BeginNs", "EndNs", "DispatchNs", "CompleteNs")


def _rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        r = list(csv.reader(f))
    if not r:
        raise ValueError(f"empty PMC csv: {path} (profiled run probably aborted)")
    head, out = r[0], []
    for row in r[1:]:
        d: dict = {}
        for c, v in zip(head, row):
            base = INST.sub("", c)
            if c in META[:1]:
                d[c] = v
                continue
            try:
                x = float(v)
            except ValueError:
                d[base] = v
                continue
            d[base] = d.get(base, 0.0) + x if INST.search(c) else x
        out.append(d)
    return out


def load(read_csv: str | Path, write_csv: str | Path | None = None) -> list[dict]:
    rd = _rows(Path(read_csv))
    wr = _rows(Path(write_csv)) if write_csv else [{}] * len(rd)
    if write_csv and len(wr) != len(rd):
        raise ValueError(f"read/write runs have different kernel counts ({len(rd)} vs {len(wr)})")
    recs = []
    for a, b in zip(rd, wr):
        if b and b.get("KernelName") and b["KernelName"] != a["KernelName"]:
            raise ValueError(f"read/write kernel order differs: {a['KernelName']} vs {b['KernelName']}")
        rdb = 64 * (a.get("TCC_EA_RDREQ", 0) - a.get("TCC_EA_RDREQ_32B", 0)) + 32 * a.get("TCC_EA_RDREQ_32B", 0) \
            + 64 * (a.get("TCC_EA1_RDREQ", 0) - a.get("TCC_EA1_RDREQ_32B", 0)) + 32 * a.get("TCC_EA1_RDREQ_32B", 0)
        wrb = (64 * b.get("TCC_EA_WRREQ_64B", 0) + 32 * (b.get("TCC_EA_WRREQ", 0) - b.get("TCC_EA_WRREQ_64B", 0))
               + 64 * b.get("TCC_EA1_WRREQ_64B", 0) + 32 * (b.get("TCC_EA1_WRREQ", 0) - b.get("TCC_EA1_WRREQ_64B", 0))) \
            if b else None
        t_ns = [x for x in (a.get("EndNs", 0) - a.get("BeginNs", 0),
                            (b.get("EndNs", 0) - b.get("BeginNs", 0)) if b else 0) if x > 0]
        recs.append({
            "name": a["KernelName"],
            "grid_threads": int(a.get("grd", 0)), "block": int(a.get("wgr", 0)),
            "ctas": int(a.get("grd", 0)) // max(1, int(a.get("wgr", 1))),
            "lds_bytes": int(a.get("lds", 0)), "scratch": int(a.get("scr", 0)),
            "arch_vgpr": int(a.get("arch_vgpr", 0)), "accum_vgpr": int(a.get("accum_vgpr", 0)),
            "sgpr": int(a.get("sgpr", 0)),
            "t_us": min(t_ns) / 1e3 if t_ns else None,          # min of the two replays
            "t_us_runs": [x / 1e3 for x in t_ns],
            "cycles": a.get("GRBM_GUI_ACTIVE", 0.0),
            "hbm_rd_bytes": rdb, "hbm_wr_bytes": wrb,
            "valu_insts": a.get("SQ_INSTS_VALU", 0.0),
            "valu_slots": a.get("SQ_ACTIVE_INST_VALU", 0.0),
            "vmem_rd_insts": a.get("SQ_INSTS_VMEM_RD", 0.0), "vmem_wr_insts": a.get("SQ_INSTS_VMEM_WR", 0.0),
            "lds_insts": a.get("SQ_INSTS_LDS", 0.0),
            "lds_bank_conflict_cycles": a.get("SQ_LDS_BANK_CONFLICT", 0.0),
            "lds_wait": a.get("SQ_WAIT_INST_LDS", 0.0),
            "ta_busy": a.get("TA_TA_BUSY", 0.0),          # summed over instances (see header)
            "ta_data_stall": a.get("TCP_TCP_TA_DATA_STALL_CYCLES", 0.0),
        })
    return recs


def window(recs: list[dict], marker: str | None) -> list[dict]:
    """Kernels after the LAST marker kernel (the measured iteration)."""
    if not marker:
        return recs
    rx = re.compile(marker)
    idx = [i for i, r in enumerate(recs) if rx.search(r["name"])]
    if not idx:
        raise ValueError(f"marker {marker!r} not found in PMC rows")
    return recs[idx[-1] + 1:]
