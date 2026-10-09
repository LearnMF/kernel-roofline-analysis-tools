"""L3: SQTT instruction-stall analysis from XCompute CLI CSV exports.

XCompute (4.6.x) attributes SQTT stall latency per opcode (`hot_inst`) and per
consecutive-instruction issue gap (`bubble_summary`); it gives no PC/source line in CLI
mode, so subtypes are derived from opcode groups.  Rules (thresholds in thresholds.json
where applicable) produce *signals*; source-level root cause still needs reading the ISA.
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

DEFAULT_BIN = "/Applications/XCompute-Professional-4.6.3.app/Contents/MacOS/XCompute-Professional-4.6.3"

GROUPS = [  # (group, regex on opcode) -- first match wins
    ("waitcnt", r"^s_waitcnt"),
    ("barrier", r"^s_barrier"),
    ("mmac", r"^v_mmac"),
    ("lds", r"^ds_"),
    ("global_load", r"^(global|buffer|flat)_load"),
    ("global_store", r"^(global|buffer|flat)_(store|atomic)"),
    ("scalar_mem", r"^s_(load|buffer_load|store)"),
    ("exec_branch", r"^s_(and|or|andn2|xor)_saveexec|^s_cbranch|^s_branch|^s_(and|or|andn2)_b64"),
    ("transcendental", r"^v_(exp|log|rcp|rsq|sqrt|sin|cos)_"),
    ("valu_int_addr", r"^v_(add|sub|addc|subb|lshl|lshr|ashr|mul_lo|mul_hi|mad_u|mad_i|and|or|xor|bfe|bfi|alignbit|perm|cndmask|lshl_add|add3|lshl_or|and_or|mov)"),
    ("valu_fp", r"^v_"),
    ("salu", r"^s_"),
]


def group_of(op: str) -> str:
    for g, rx in GROUPS:
        if re.search(rx, op):
            return g
    return "other"


def export(perf: str | Path, outdir: str | Path, dispatches: str | None = None,
           sections: str = "dispatch_summary,hot_inst,bubble_summary", top: int = 400,
           xbin: str | None = None) -> Path:
    xbin = xbin or os.environ.get("XCOMPUTE_BIN", DEFAULT_BIN)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [xbin, "-P", str(perf), "--sqtt-sections", sections, "--sqtt-top", str(top),
           "-F", "csv", "-D", str(outdir)]
    if dispatches:
        cmd += ["--sqtt-dispatches", dispatches]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return outdir


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def summarize(csv_dir: str | Path) -> dict:
    d = Path(csv_dir)
    find = lambda sfx: next(iter(sorted(d.glob(f"*_sqtt_{sfx}.csv"))), Path("/nonexistent"))
    disp = {r["Dispatch id"]: r for r in _rows(find("dispatch_summary"))}
    out: dict[str, dict] = {}
    for r in _rows(find("hot_inst")):
        k = out.setdefault(r["Dispatch id"], {"kernel": r["Kernel"], "lat": defaultdict(float),
                                              "ops": defaultdict(float), "bub": defaultdict(float)})
        lat = float(r["Latency"])
        k["lat"][group_of(r["Opcode"])] += lat
        k["ops"][r["Opcode"]] += lat
    for r in _rows(find("bubble_summary")):
        k = out.get(r["Dispatch id"])
        if k is None:
            continue
        key = f"{group_of(r['Before'])}->{group_of(r['After'])}"
        k["bub"][key] += float(r["Duration"])
    res = {}
    for did, k in out.items():
        tot = sum(k["lat"].values()) or 1.0
        btot = sum(k["bub"].values()) or 1.0
        share = {g: v / tot for g, v in sorted(k["lat"].items(), key=lambda kv: -kv[1])}
        bshare = {g: v / btot for g, v in sorted(k["bub"].items(), key=lambda kv: -kv[1])[:8]}
        ds = disp.get(did, {})
        sig = signals(share, bshare)
        res[did] = {"kernel": k["kernel"], "latency_share": share, "bubble_share_top": bshare,
                    "top_opcodes": [(o, v / tot) for o, v in sorted(k["ops"].items(), key=lambda kv: -kv[1])[:8]],
                    "avg_active_waves": float(ds.get("Avg Active Waves", 0) or 0),
                    "idle_pct": float(ds.get("Idle %", 0) or 0), "signals": sig}
    return res


def signals(share: dict, bshare: dict) -> list[dict]:
    s = lambda *gs: sum(share.get(g, 0.0) for g in gs)
    b = lambda pfx: sum(v for k, v in bshare.items() if k.startswith(pfx))
    out = []
    if s("waitcnt") >= 0.30:
        out.append({"id": "C.wait_exposed", "value": round(s("waitcnt"), 3),
                    "note": "s_waitcnt dominates: memory (vmcnt) or LDS/scalar (lgkmcnt) latency exposed; "
                            "split by reading the waitcnt sites in the ISA"})
    if s("barrier") + b("barrier->") >= 0.20:
        out.append({"id": "C.sync", "value": round(s("barrier") + b("barrier->"), 3),
                    "note": "time after s_barrier: CTA-level synchronization / imbalance between waves"})
    if s("valu_int_addr", "salu") >= 0.20:
        out.append({"id": "D.addr_overhead", "value": round(s("valu_int_addr", "salu"), 3),
                    "note": "integer/address arithmetic and scalar ops take a large share of issue latency"})
    if s("exec_branch") >= 0.10:
        out.append({"id": "C.divergence", "value": round(s("exec_branch"), 3),
                    "note": "exec-mask / branch instructions"})
    if s("global_load", "global_store") >= 0.25:
        out.append({"id": "C.vmem_issue", "value": round(s("global_load", "global_store"), 3),
                    "note": "stalls at vector-memory issue (queue full / address dependency)"})
    if s("lds") >= 0.15:
        out.append({"id": "C.lds_issue", "value": round(s("lds"), 3), "note": "stalls at LDS instruction issue"})
    return out


def to_markdown(res: dict, title: str = "SQTT (L3)") -> str:
    L = [f"# {title}", "", "| dispatch | kernel | top latency groups | top bubbles (before->after) | signals |",
         "|---|---|---|---|---|"]
    from ..timeline.analyze import short_name
    for did, r in sorted(res.items(), key=lambda kv: int(kv[0])):
        top = ", ".join(f"{g} {v:.0%}" for g, v in list(r["latency_share"].items())[:5])
        bub = ", ".join(f"{g} {v:.0%}" for g, v in list(r["bubble_share_top"].items())[:3])
        sig = "<br>".join(f"`{x['id']}` {x['value']:.0%}" for x in r["signals"]) or "—"
        L.append(f"| {did} | {short_name(r['kernel'])} | {top} | {bub} | {sig} |")
    return "\n".join(L) + "\n"
