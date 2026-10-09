"""L1 system-level analysis: where does the operator's wall time go?

For each measurement window (one operator invocation, cut by marker kernels):
  * span        first device-op begin -> last device-op end
  * busy        union of device-op intervals (all streams)
  * bubble      span - busy ; bubble_ratio = bubble / span
  * gaps        every idle interval, attributed to a cause:
      sync        a host synchronize/blocking API ended inside the gap
      host_late   the next op was submitted by the host after the previous op had
                  finished -> the host could not keep the queue fed (launch/framework
                  bound; hipGraph or fewer launches recover it)
      dispatch    the next op was already queued; the gap is device-side dispatch
                  latency between dependent kernels (fusion/persistent kernels recover
                  it; graphs only partially)
      unknown     no submit timestamp in the trace
  * per-kernel  count / total / share of span, small-kernel count
  * A-class verdict against kra/thresholds.json, plus the launch lower bound
    n_kernels * launch_gap_min when a machine.json is given.
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

from .hipprof_json import DeviceOp, Trace

SYNC_API = re.compile(r"Synchronize|hipMemcpy(WithStream)?$|hipMemcpyDtoH$|hipMemcpyHtoD$|"
                      r"hipStreamWaitEvent|hipEventSynchronize|hipStreamQuery")

_THRESH_PATH = Path(__file__).resolve().parent.parent / "thresholds.json"


def thresholds() -> dict:
    t = json.loads(_THRESH_PATH.read_text())["thresholds"]
    return {k: v["value"] for k, v in t.items()}


def _itanium_last_component(name: str) -> str | None:
    """Last component of an Itanium-mangled nested name (_ZN<len><id>...[I...]E...),
    e.g. _ZN2g212_GLOBAL__N_119g2_l2n_apply_kernelEPKs -> g2_l2n_apply_kernel."""
    m = re.match(r"_ZN?((?:[0-9]+[A-Za-z_][A-Za-z0-9_]*)+)", name)
    if not m:
        return None
    parts, s, i = [], m.group(1), 0
    while i < len(s) and s[i].isdigit():
        j = i
        while j < len(s) and s[j].isdigit():
            j += 1
        n = int(s[i:j])
        parts.append(s[j:j + n])
        i = j + n
    return parts[-1] if parts else None


def short_name(name: str) -> str:
    """Demangled-ish kernel name -> compact label (template args and params dropped)."""
    if name.startswith("_Z"):
        comp = _itanium_last_component(name)
        if comp:
            return comp
    n = re.sub(r"\(anonymous namespace\)::", "", name)
    n = re.sub(r"\(.*$", "", n)                    # parameter list
    n = re.sub(r"^void\s+", "", n)
    while re.search(r"<[^<>]*>", n):
        n = re.sub(r"<[^<>]*>", "", n)             # template arguments
    return n.split("::")[-1].strip() or name[:60]


def split_windows(tr: Trace, marker: str | None, device: int | None) -> list[list[DeviceOp]]:
    ops = [o for o in tr.ops if device is None or o.device == device]
    if not marker:
        return [ops] if ops else []
    rx = re.compile(marker)
    marks = [i for i, o in enumerate(ops) if o.kind == "kernel" and rx.search(o.name)]
    wins = []
    for k, i in enumerate(marks):
        j = marks[k + 1] if k + 1 < len(marks) else len(ops)
        w = ops[i + 1:j]
        if w:
            wins.append(w)
    return wins


def _union(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for b, e in sorted(intervals):
        if out and b <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([b, e])
    return [(b, e) for b, e in out]


def analyze_window(ops: list[DeviceOp], apis, th: dict, machine: dict | None) -> dict:
    begin, end = min(o.begin_ns for o in ops), max(o.end_ns for o in ops)
    span = end - begin
    busy_iv = _union([(o.begin_ns, o.end_ns) for o in ops])
    busy = sum(e - b for b, e in busy_iv)
    kernels = [o for o in ops if o.kind == "kernel"]
    k_sum = sum(o.dur_ns for o in kernels)
    c_sum = sum(o.dur_ns for o in ops if o.kind == "copy")

    # gaps between consecutive busy intervals; attribute using the op that ends the
    # previous interval and the op that starts the next one.
    gaps = []
    by_begin = sorted(ops, key=lambda o: o.begin_ns)
    for (b0, e0), (b1, e1) in zip(busy_iv, busy_iv[1:]):
        prev = max((o for o in ops if o.end_ns <= e0 and o.end_ns > b0 - 1), key=lambda o: o.end_ns)
        nxt = next(o for o in by_begin if o.begin_ns >= b1)
        g = b1 - e0
        sync = [a for a in apis if SYNC_API.search(a.name) and e0 <= a.end_ns <= b1 + 1000]
        if sync:
            cause = "sync"
        elif nxt.submit_ns is None:
            cause = "unknown"
        elif nxt.submit_ns > e0:
            cause = "host_late"
        else:
            cause = "dispatch"
        gaps.append({
            "gap_us": g / 1e3, "after": short_name(prev.name), "before": short_name(nxt.name),
            "cause": cause,
            "submit_lead_us": None if nxt.submit_ns is None else (e0 - nxt.submit_ns) / 1e3,
            "sync_api": sync[0].name if sync else None,
        })

    per_kernel: dict[str, dict] = {}
    for o in kernels:
        s = per_kernel.setdefault(short_name(o.name), {"count": 0, "total_us": 0.0, "full_name": o.name})
        s["count"] += 1
        s["total_us"] += o.dur_ns / 1e3
    for s in per_kernel.values():
        s["mean_us"] = s["total_us"] / s["count"]
        s["share_of_span"] = s["total_us"] * 1e3 / span if span else 0.0
    per_kernel = dict(sorted(per_kernel.items(), key=lambda kv: -kv[1]["total_us"]))

    streams: dict[str, int] = {}
    for o in ops:
        streams[o.stream] = streams.get(o.stream, 0)
    for s in streams:
        streams[s] = sum(e - b for b, e in _union([(o.begin_ns, o.end_ns) for o in ops if o.stream == s]))

    by_cause: dict[str, float] = {}
    for g in gaps:
        by_cause[g["cause"]] = by_cause.get(g["cause"], 0.0) + g["gap_us"]

    small = [o for o in kernels if o.dur_ns / 1e3 < th["system.small_kernel_us"]]
    # Ops that were already queued when their predecessor finished: hipprof reports
    # them back-to-back (gap 0), with the dispatch overhead folded into the duration.
    seq = sorted(ops, key=lambda o: o.begin_ns)
    queued = sum(1 for a, b in zip(seq, seq[1:])
                 if b.begin_ns - a.end_ns <= 0 and b.submit_ns is not None and b.submit_ns <= a.end_ns)
    res = {
        "span_us": span / 1e3,
        "busy_us": busy / 1e3,
        "bubble_us": (span - busy) / 1e3,
        "bubble_ratio": (span - busy) / span if span else 0.0,
        "kernel_sum_us": k_sum / 1e3,
        "copy_sum_us": c_sum / 1e3,
        "concurrency": (k_sum + c_sum) / busy if busy else 0.0,
        "n_kernels": len(kernels),
        "n_copies": len(ops) - len(kernels),
        "n_small_kernels": len(small),
        "n_queued_back_to_back": queued,
        "n_gaps": len(gaps),
        "gap_us_by_cause": by_cause,
        "gap_us_median": statistics.median([g["gap_us"] for g in gaps]) if gaps else 0.0,
        "largest_gaps": sorted(gaps, key=lambda g: -g["gap_us"])[:10],
        "streams_busy_us": {s: v / 1e3 for s, v in streams.items()},
        "per_kernel": per_kernel,
    }
    if machine:
        launch = machine.get("launch", {})
        lg = (launch.get("launch_graph") or {}).get("best_us")
        ls = (launch.get("launch_stream") or {}).get("best_us")
        res["launch_floor_us"] = {
            "stream": len(ops) * ls if ls else None,
            "graph": len(ops) * lg if lg else None,
            "note": "n_ops x back-to-back empty-kernel cost from machine.json (dispatch floor)",
        }
        if lg:
            # Bubble a graph/fusion could recover at most: gap time above the per-op floor.
            res["recoverable_bubble_us_upper"] = sum(
                max(0.0, g["gap_us"] - lg) for g in gaps if g["cause"] in ("host_late", "dispatch"))
        if ls:
            # Dispatch overhead hidden inside back-to-back durations (unprofiled cost per
            # dependent dispatch, measured by L0; upper bound: includes the ~0.5 us body
            # of the empty calibration kernel).
            res["embedded_dispatch_us_upper"] = queued * ls
            res["embedded_dispatch_ratio_upper"] = queued * ls * 1e3 / span if span else 0.0
    return res


def verdict(win: dict, th: dict) -> dict:
    """Layer-A classification for one window (see kra/taxonomy.json A.*)."""
    flags = []
    br = win["bubble_ratio"]
    if br >= th["system.bubble_ratio_investigate"]:
        dom = max(win["gap_us_by_cause"].items(), key=lambda kv: kv[1])[0] if win["gap_us_by_cause"] else None
        cls = {"host_late": "A.launch", "dispatch": "A.launch", "sync": "A.stream_dep"}.get(dom, "A.launch")
        flags.append({"class": cls, "reason": f"bubble_ratio {br:.1%} >= {th['system.bubble_ratio_investigate']:.0%}; dominant gap cause: {dom}"})
    if win["n_kernels"] and win["n_small_kernels"] / win["n_kernels"] > 0.5:
        flags.append({"class": "E.tiny", "reason": f"{win['n_small_kernels']}/{win['n_kernels']} kernels shorter than {th['system.small_kernel_us']} us"})
    copy_share = win["copy_sum_us"] / win["span_us"] if win["span_us"] else 0
    if copy_share > 0.15:
        flags.append({"class": "A.transfer", "reason": f"copies/memsets occupy {copy_share:.1%} of span"})
    edr = win.get("embedded_dispatch_ratio_upper")
    if edr is not None and edr >= th["system.bubble_ratio_investigate"]:
        flags.append({"class": "A.launch", "reason": f"hidden dispatch overhead of {win['n_queued_back_to_back']} dependent launches <= {edr:.1%} of span (fusion / persistent kernel)"})
    if len(win["streams_busy_us"]) > 1:
        flags.append({"class": "A.stream_dep", "reason": "multiple streams active: check overlap/critical path (inter-stream dependencies are not recorded in the hipprof trace)", "informational": True})
    return {
        "system_bound": any(f["class"].startswith("A.") and not f.get("informational") for f in flags),
        "flags": flags,
        "next_step": ("fix system-level bubbles first (Graph / fewer launches / remove syncs), then per-kernel analysis"
                      if br >= th["system.bubble_ratio_investigate"] else
                      "bubbles below threshold: time is spent inside kernels -> proceed to per-kernel classification (L2)"),
    }


def _agg(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def analyze(tr: Trace, marker: str | None = None, device: int | None = None,
            machine: dict | None = None, reference_ms: float | None = None) -> dict:
    th = thresholds()
    wins = split_windows(tr, marker, device)
    if not wins:
        raise ValueError("no device ops found (check --marker / --device)")
    results = [analyze_window(w, tr.apis, th, machine) for w in wins]
    for r in results:
        r["verdict"] = verdict(r, th)
    keys = ("span_us", "busy_us", "bubble_us", "bubble_ratio", "kernel_sum_us", "n_kernels", "gap_us_median",
            "embedded_dispatch_ratio_upper")
    summary = {k: _agg([r.get(k) for r in results]) for k in keys}
    trust = {"reference_ms": reference_ms}
    if reference_ms:
        ratio = summary["span_us"]["median"] / (reference_ms * 1e3)
        trust["traced_over_reference"] = ratio
        trust["ok"] = ratio <= th["trust.profiler_vs_bench_ratio_max"]
        trust["note"] = ("traced span vs. unprofiled per-iteration time; tracing inflates host API cost, "
                         "so host_late gaps are upper bounds when this ratio is > 1")
    # Representative window = the one with the median span.
    rep = sorted(results, key=lambda r: r["span_us"])[len(results) // 2]
    return {
        "schema": "kra.timeline/1",
        "source": tr.source,
        "marker": marker,
        "device": device,
        "n_windows": len(results),
        "trust": trust,
        "summary": summary,
        "representative": rep,
        "windows": [{k: v for k, v in r.items() if k not in ("per_kernel", "largest_gaps")} for r in results],
    }


def to_markdown(res: dict, title: str = "Timeline (L1)") -> str:
    s, rep = res["summary"], res["representative"]
    L = [f"# {title}", "",
         f"source: `{res['source']}` | windows: {res['n_windows']} | marker: `{res['marker']}`", ""]
    if res["trust"].get("reference_ms"):
        t = res["trust"]
        L += [f"**Trust check**: traced span / unprofiled = {t['traced_over_reference']:.2f} "
              f"({'OK' if t['ok'] else 'SUSPECT'}) — {t['note']}", ""]
    L += ["| metric | median | min | max |", "|---|---:|---:|---:|"]
    for k in ("span_us", "busy_us", "bubble_us", "bubble_ratio", "kernel_sum_us", "n_kernels", "gap_us_median",
              "embedded_dispatch_ratio_upper"):
        a = s.get(k)
        if a is None:
            continue
        fmt = (lambda v: f"{v:.2%}") if "ratio" in k else (lambda v: f"{v:,.1f}")
        L.append(f"| {k} | {fmt(a['median'])} | {fmt(a['min'])} | {fmt(a['max'])} |")
    v = rep["verdict"]
    L += ["", "## Verdict (representative window)", "",
          f"- system bound: **{v['system_bound']}**"]
    L += [f"- `{f['class']}`: {f['reason']}" for f in v["flags"]] or ["- no layer-A flags"]
    L += [f"- next: {v['next_step']}", ""]
    L += ["## Gap attribution (representative window)", "",
          "| cause | total us | share of span |", "|---|---:|---:|"]
    for c, us in sorted(rep["gap_us_by_cause"].items(), key=lambda kv: -kv[1]):
        L.append(f"| {c} | {us:,.1f} | {us * 1e3 / (rep['span_us'] * 1e3):.2%} |")
    if "launch_floor_us" in rep:
        lf = rep["launch_floor_us"]
        L += ["", f"- launch floor (n_ops × empty-kernel cost, L0): stream {lf['stream']:.1f} us, graph {lf['graph']:.1f} us",
              f"- visible bubble recoverable by graph/fusion (upper bound): {rep.get('recoverable_bubble_us_upper', 0):.1f} us",
              f"- dispatch overhead hidden inside {rep['n_queued_back_to_back']} back-to-back durations (upper bound): "
              f"{rep.get('embedded_dispatch_us_upper', 0):.1f} us = {rep.get('embedded_dispatch_ratio_upper', 0):.2%} of span "
              "(hipprof folds dependent-dispatch cost into kernel durations, see docs/methodology.md)"]
    L += ["", "### Largest gaps", "",
          "submit lead = previous op end − next op host submit time (> 0: already queued; < 0: host submitted late)", "",
          "| gap us | after | before | cause | submit lead us |", "|---:|---|---|---|---:|"]
    for g in rep["largest_gaps"]:
        lead = "" if g["submit_lead_us"] is None else f"{g['submit_lead_us']:.1f}"
        L.append(f"| {g['gap_us']:.1f} | {g['after']} | {g['before']} | {g['cause']} | {lead} |")
    L += ["", "## Kernels (representative window)", "",
          "| kernel | count | total us | mean us | share of span |", "|---|---:|---:|---:|---:|"]
    for n, k in rep["per_kernel"].items():
        L.append(f"| {n} | {k['count']} | {k['total_us']:,.1f} | {k['mean_us']:,.1f} | {k['share_of_span']:.1%} |")
    return "\n".join(L) + "\n"
