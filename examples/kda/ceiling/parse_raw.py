"""Merge raw per-shape captures into a per-kernel metrics table.

Inputs per shape dir raw/<T>_<H>/:
  trace_1..5.json     hipprof --hip-trace (iters 6, KRA_JITTER_SEED=1..5) -> device times
  pmc_read/write.csv  hipprof presets (TA/VALU/VMEM/LDS insts, EA bytes, static info)
  pmc_l2.csv          custom counters TCC_REQ/READ/WRITE/HIT_sum
  pmc_sq.csv          custom counters SQ_INSTS_MMOP, SQ_ACTIVE_INST_MMOP, SQ_LDS_IDX_ACTIVE
  optrace.json        launcher interface bytes (required-HBM-bytes denominator)
Usage:  rows = parse_shape("raw/16384_48")   (also used by cards.py)
"""
from __future__ import annotations

import csv
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from kra.timeline.hipprof_json import load as tload          # noqa: E402
from kra.timeline.analyze import split_windows, short_name  # noqa: E402


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def short(n: str) -> str:
    n = short_name(n)
    n = n.replace("g2_", "").replace("p3_", "").replace("_kernel", "")
    return n


# ---------------------------------------------------------------- trace -> per-kernel times
def trace_times(d: Path) -> dict:
    """name#occ -> {median_us (of per-process medians), spread, per_proc[]}"""
    per_kernel: dict[str, list[list[float]]] = {}
    for i in range(1, 6):
        p = d / f"trace_{i}.json"
        if not p.exists():
            continue
        cur: dict[str, list[float]] = {}
        for w in split_windows(tload(p), "spin_kernel", None):
            occ: dict[str, int] = {}
            for o in w:
                if o.kind != "kernel":
                    continue
                n = short(o.name)
                occ[n] = occ.get(n, 0) + 1
                cur.setdefault(f"{n}#{occ[n]}", []).append(o.dur_ns / 1e3)
        for k, v in cur.items():
            per_kernel.setdefault(k, []).append(v)
    out = {}
    for k, procs in per_kernel.items():
        meds = [st.median(v) for v in procs]
        med = st.median(meds)
        out[k] = dict(t_us=med, spread=(max(meds) - min(meds)) / med if med else 0.0,
                      procs=len(meds), spread_all=meds)
    return out


# ---------------------------------------------------------------- PMC csvs -> counters
def _csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return list(csv.DictReader(open(path)))


def _window(rows: list[dict]) -> list[dict]:
    idx = [i for i, r in enumerate(rows) if "spin_kernel" in (r.get("KernelName") or "")]
    return rows[idx[-1] + 1:] if idx else rows


def _inst_sum(row: dict, base: str) -> float:
    """Sum counter columns including [i] instance suffixes."""
    tot = 0.0
    for k, v in row.items():
        b = re.sub(r"\[\d+\]$", "", k)
        if b == base:
            tot += _f(v)
    return tot


def pmc_rows(d: Path) -> dict:
    """name#occ -> merged counters from the 4 passes (rows joined by order)."""
    rd, wr = _window(_csv(d / "pmc_read.csv")), _window(_csv(d / "pmc_write.csv"))
    l2, sq = _window(_csv(d / "pmc_l2.csv")), _window(_csv(d / "pmc_sq.csv"))
    n = min(len(rd), len(l2), len(sq))
    out, occ = {}, {}
    for i in range(n):
        a, b, c, e = rd[i], wr[i] if i < len(wr) else {}, l2[i], sq[i]
        nm = short(a.get("KernelName", ""))
        occ[nm] = occ.get(nm, 0) + 1
        key = f"{nm}#{occ[nm]}"
        rec = dict(
            name=nm,
            ctas=int(_f(a.get("grd")) // max(1, _f(a.get("wgr")))),
            block=int(_f(a.get("wgr"))),
            lds_c=int(_f(a.get("lds"))), scr=int(_f(a.get("scr"))),
            vgpr=int(_f(a.get("arch_vgpr"))), sgpr=int(_f(a.get("sgpr"))),
            grbm=_inst_sum(a, "GRBM_GUI_ACTIVE"),
            ta_busy=_inst_sum(a, "TA_TA_BUSY"),
            ta_rd_wf=_inst_sum(a, "TA_FLAT_READ_WAVEFRONTS"),
            ta_wr_wf=_inst_sum(a, "TA_FLAT_WRITE_WAVEFRONTS"),
            valu_insts=_inst_sum(a, "SQ_INSTS_VALU"),
            valu_slots=_inst_sum(a, "SQ_ACTIVE_INST_VALU"),
            vmem_rd=_inst_sum(a, "SQ_INSTS_VMEM_RD"),
            vmem_wr=_inst_sum(a, "SQ_INSTS_VMEM_WR"),
            lds_insts=_inst_sum(a, "SQ_INSTS_LDS"),
            lds_wait=_inst_sum(a, "SQ_WAIT_INST_LDS"),
            lds_conf=_inst_sum(a, "SQ_LDS_BANK_CONFLICT"),
            mmop=_inst_sum(e, "SQ_INSTS_MMOP"),
            mmop_active=_inst_sum(e, "SQ_ACTIVE_INST_MMOP"),
            lds_idx=_inst_sum(e, "SQ_LDS_IDX_ACTIVE"),
            tcc_req=_inst_sum(c, "TCC_REQ_sum") or _inst_sum(c, "TCC_REQ"),
            tcc_rd=_inst_sum(c, "TCC_READ_sum") or _inst_sum(c, "TCC_READ"),
            tcc_wr=_inst_sum(c, "TCC_WRITE_sum") or _inst_sum(c, "TCC_WRITE"),
            tcc_hit=_inst_sum(c, "TCC_HIT_sum") or _inst_sum(c, "TCC_HIT"),
            hbm_rd=64 * (_inst_sum(a, "TCC_EA_RDREQ") - _inst_sum(a, "TCC_EA_RDREQ_32B"))
                   + 32 * _inst_sum(a, "TCC_EA_RDREQ_32B"),
            hbm_wr=64 * _inst_sum(b, "TCC_EA_WRREQ_64B")
                   + 32 * (_inst_sum(b, "TCC_EA_WRREQ") - _inst_sum(b, "TCC_EA_WRREQ_64B")),
        )
        out[key] = rec
    return out


def optrace_launchers(d: Path) -> dict:
    j = json.load(open(d / "optrace.json")) if (d / "optrace.json").exists() else {}
    return {c["launcher"]: c for c in j.get("calls", [])}


def parse_shape(d: str | Path) -> dict:
    d = Path(d)
    t = trace_times(d)
    p = pmc_rows(d)
    for k, rec in p.items():
        if k in t:
            rec["t_us"] = t[k]["t_us"]
            rec["spread"] = t[k]["spread"]
            rec["procs"] = t[k]["procs"]
    return dict(kernels=p, times=t, launchers=optrace_launchers(d))


if __name__ == "__main__":
    import sys
    r = parse_shape(sys.argv[1])
    for k, v in sorted(r["kernels"].items(), key=lambda x: -x[1].get("t_us", 0)):
        print(f"{k:28s} t={v.get('t_us', 0):9.1f}us ctas={v['ctas']:4d} "
              f"hbm={(v['hbm_rd'] + v['hbm_wr']) / 1e6:8.0f}MB l2={v['tcc_req'] * 64 / 1e6:8.0f}MB "
              f"mmop={v['mmop']:12.0f} ta={v['ta_busy'] / max(1, v['grbm']) :5.2f} "
              f"valu={v['valu_slots'] / (v['grbm'] * 80):5.2f}")
