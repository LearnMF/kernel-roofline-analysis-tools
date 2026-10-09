"""L3+: per-instruction (PC-level) stall attribution from an XCompute `pipeline` dump.

XCompute's CLI has no PC column, but `--sqtt-sections pipeline --sqtt-location ...
--sqtt-time-range a:b` emits one wave's full issue stream (inst + bubble events with
timestamps).  Inside the hot loop control flow repeats, so each iteration's dynamic
opcode sequence can be aligned to the static loop body of the kernel's ISA (compiled with
-gline-tables-only: identical codegen, plus .loc source lines).  The bubble that precedes
an instruction is the time the wave waited to issue it, i.e. the stall attributed to it.
"""
from __future__ import annotations

import csv
import difflib
import re
from collections import defaultdict
from pathlib import Path

from ..isa.scan import BRANCH, INSN, LABEL

FILE = re.compile(r'^\s*\.file\s+(\d+)\s+"([^"]*)"(?:\s+"([^"]*)")?')
LOC = re.compile(r"^\s*\.loc\s+(\d+)\s+(\d+)")


def static_body(asm: str, symbol: str) -> tuple[list[dict], list[tuple[int, int]]]:
    files, insns, labels, branches = {}, [], {}, []
    cur, on = (None, 0), False
    for ln in asm.splitlines():
        if (m := FILE.match(ln)):
            files[m.group(1)] = m.group(3) or m.group(2)
        if not on:
            on = ln.startswith(symbol + ":")
            continue
        if ln.startswith(".Lfunc_end"):
            break
        if (m := LOC.match(ln)):
            cur = (m.group(1), int(m.group(2)))
            continue
        if (m := LABEL.match(ln)):
            labels[m.group(1)] = len(insns)
            continue
        if (m := INSN.match(ln)):
            if (b := BRANCH.match(ln)):
                branches.append((len(insns), b.group(1)))
            src = f"{Path(files.get(cur[0], '?')).name}:{cur[1]}" if cur[0] else "?"
            insns.append({"op": m.group(1), "text": ln.split(";")[0].strip(), "src": src})
    loops = [(labels[t], i) for i, t in branches if t in labels and labels[t] <= i]
    mm = lambda a, b: sum(1 for x in insns[a:b + 1] if x["op"].startswith("v_mmac"))
    cand = [ab for ab in loops if mm(*ab) > 0]
    hot = [ab for ab in cand if not any(c != ab and ab[0] <= c[0] and c[1] <= ab[1] for c in cand)]
    return insns, hot


def dynamic_stream(pipeline_csv: str | Path, wf: str | None = None) -> list[dict]:
    rows = list(csv.DictReader(open(pipeline_csv, newline="")))
    if wf is None:
        wf = rows[0]["WF ID"]
    rows = sorted((r for r in rows if r["WF ID"] == wf), key=lambda r: (int(r["Start"]), r["Kind"] != "bubble"))
    # A gap right after s_barrier / s_waitcnt / s_sleep is the wave WAITING at that
    # instruction (barrier arrival, memory counters) -> charged to it ("wait").  Any other
    # gap is the next instruction failing to issue (dependency, structural/queue full)
    # -> charged to the next instruction ("stall").
    out, pending = [], 0
    for r in rows:
        if r["Kind"] == "bubble":
            if out and r.get("Before", "").startswith(("s_barrier", "s_waitcnt", "s_sleep")):
                out[-1]["wait"] += int(r["Duration"])
            else:
                pending += int(r["Duration"])
        elif r["Kind"] == "inst":
            out.append({"op": r["Opcode"], "issue": int(r["Issue TS"] or r["Start"]), "stall": pending,
                        "wait": 0, "coissue": int(r.get("CoIssue Width") or 1)})
            pending = 0
    for d in out:
        d["stall"] += d.pop("wait")
    return out


def attribute(insns: list[dict], loop: tuple[int, int], dyn: list[dict], k: int = 8) -> dict:
    body = [x["op"] for x in insns[loop[0]:loop[1] + 1]]
    dops = [d["op"] for d in dyn]
    starts = [i for i in range(len(dops) - k) if dops[i:i + k] == body[:k]]
    per = defaultdict(lambda: [0, 0])          # static idx -> [stall sum, hits]
    steps, mapped, total = 0, 0, 0
    for s, e in zip(starts, starts[1:]):
        seg = dyn[s:e]
        sm = difflib.SequenceMatcher(None, body, [d["op"] for d in seg], autojunk=False)
        for blk in sm.get_matching_blocks():
            for t in range(blk.size):
                d = seg[blk.b + t]
                per[loop[0] + blk.a + t][0] += d["stall"]
                per[loop[0] + blk.a + t][1] += 1
                mapped += d["stall"]
        total += sum(d["stall"] for d in seg) + len(seg)
        steps += 1
    cyc_per_step = (dyn[starts[-1]]["issue"] - dyn[starts[0]]["issue"]) / max(1, steps) if len(starts) > 1 else 0
    rows = []
    for idx in range(loop[0], loop[1] + 1):
        st, hits = per.get(idx, [0, 0])
        rows.append({"idx": idx, "op": insns[idx]["op"], "text": insns[idx]["text"], "src": insns[idx]["src"],
                     "stall_per_step": st / max(1, steps), "hits_per_step": hits / max(1, steps)})
    by_src = defaultdict(float)
    for r in rows:
        by_src[r["src"]] += r["stall_per_step"]
    return {"steps": steps, "cycles_per_step": cyc_per_step, "body_len": len(body),
            "stall_mapped_frac": mapped / max(1, sum(d["stall"] for s, e in zip(starts, starts[1:]) for d in dyn[s:e])),
            "insts_per_step": (starts[-1] - starts[0]) / max(1, steps) if len(starts) > 1 else 0,
            "rows": rows, "by_src": dict(sorted(by_src.items(), key=lambda kv: -kv[1]))}


def report(res: dict, top: int = 25) -> str:
    cps = res["cycles_per_step"] or 1
    L = [f"steps aligned: {res['steps']}, cycles/step {res['cycles_per_step']:.0f}, "
         f"dyn insts/step {res['insts_per_step']:.0f} (static body {res['body_len']}), "
         f"stall mapped {res['stall_mapped_frac']:.0%}", "",
         "top stalled instructions (stall cycles before issue, per step):"]
    for r in sorted(res["rows"], key=lambda r: -r["stall_per_step"])[:top]:
        L.append(f"  {r['stall_per_step']:7.1f} ({r['stall_per_step']/cps:5.1%})  {r['src']:24s} {r['text'][:70]}")
    L += ["", "stall by source line (per step):"]
    for s, v in list(res["by_src"].items())[:top]:
        L.append(f"  {v:7.1f} ({v/cps:5.1%})  {s}")
    return "\n".join(L)
