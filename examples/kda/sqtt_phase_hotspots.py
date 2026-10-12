"""sqtt_hotspots restricted to one phase of each wave: instructions before (phase=A) or from (phase=B)
the N-th occurrence of a marker instruction (default: the 5th `buffer_load_dwordx4 ... lds`, the start
of the fused s4ab's phase B).  Same categories as sqtt_hotspots.py (issue / waitcnt / barrier /
issue_delay) plus the top wait sites and the issue-delay split by instruction type.
    python3 sqtt_phase_hotspots.py <dir> --phase A [--nth 5] [--marker lds] [--top 20]"""
import argparse
import collections
import glob
import json
import re

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--phase", default="A")
ap.add_argument("--marker", default="lds")
ap.add_argument("--prefix", default="buffer_load_dwordx4")
ap.add_argument("--nth", type=int, default=5)
ap.add_argument("--top", type=int, default=20)
ap.add_argument("--dispatch", default="0")
a = ap.parse_args()
wcnt = re.compile(r"(vmcnt|lgkmcnt|expcnt)\((\d+)\)")
tot = collections.Counter(); pre = collections.Counter(); site = collections.Counter(); site_n = collections.Counter()
types = collections.Counter(); waves = 0
for f in sorted(glob.glob(f"{a.dir}/*thread_trace_{a.dispatch}__*.json")):
    ev = json.load(open(f))["traceEvents"]
    streams = collections.defaultdict(list)
    for e in ev:
        if e.get("ph") == "X" and "assembly" in e.get("args", {}):
            streams[(e["pid"], e["tid"])].append(e)
    for s in streams.values():
        s.sort(key=lambda e: (e["ts"], e["args"].get("inst idx", 0)))
        cur = []
        for e in s:
            cur.append(e)
            if e["name"] != "s_endpgm":
                continue
            w, cur = cur, []
            hits = [k for k, x in enumerate(w) if x["args"]["assembly"].startswith(a.prefix) and a.marker in x["args"]["assembly"]]
            if len(hits) < a.nth:
                continue
            cut = hits[a.nth - 1]
            rng = range(1, cut) if a.phase == "A" else range(cut + 1, len(w))
            waves += 1
            for k in rng:
                x, p = w[k], w[k - 1]
                t = x["args"].get("inst type", "")
                clk = x["args"].get("clks", 4)
                types[t] += 1
                tot["issue"] += clk
                gap = max(0, x["ts"] - p["ts"] - p["args"].get("clks", 4))
                key = (x["args"]["assembly"], p["args"]["assembly"])
                if x["name"] == "s_waitcnt":
                    tot["waitcnt"] += gap; site[key] += gap; site_n[key] += 1
                elif x["name"] == "s_barrier":
                    tot["barrier"] += gap; site[key] += gap; site_n[key] += 1
                else:
                    tot["issue_delay"] += gap; pre[t] += gap
allc = sum(tot.values())
print(f"phase {a.phase}: {waves} waves, {allc / max(waves, 1):.0f} cycles per wave")
for k in ("issue", "waitcnt", "barrier", "issue_delay"):
    print(f"  {k:12s} {tot[k] / max(waves, 1):9.0f} cycles/wave  {tot[k] / allc:6.1%}")
print(f"\ntop wait sites (cycles per wave, count per wave): wait <- previous instruction")
for (k, v) in site.most_common(a.top):
    print(f"  {v / waves:8.0f}  x{site_n[k] / waves:5.1f}  {k[0][:48]:48s} <- {k[1][:60]}")
print("\nissue delay by type (cycles/wave):", {t: round(v / waves) for t, v in pre.most_common(8)})
print("instructions by type (per wave):", {t: round(v / waves) for t, v in types.most_common(10)})
