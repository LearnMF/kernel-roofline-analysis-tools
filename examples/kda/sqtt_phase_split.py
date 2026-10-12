"""Per-wave phase split from hipprof SQTT traces: wave time before / after the N-th occurrence of a
marker instruction (e.g. the 5th `buffer_load_dwordx4 ... lds` = start of the fused s4ab's phase B).
    python3 sqtt_phase_split.py <dir> --marker "lds" --nth 5 [--dispatch 0]"""
import argparse
import collections
import glob
import json
import statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--marker", default="lds")
ap.add_argument("--prefix", default="buffer_load_dwordx4")
ap.add_argument("--nth", type=int, default=5)
ap.add_argument("--dispatch", default="0")
a = ap.parse_args()
pa, pb = [], []
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
            hits = [x for x in w if x["args"]["assembly"].startswith(a.prefix) and a.marker in x["args"]["assembly"]]
            if len(hits) < a.nth:
                continue
            t0, tm, t1 = w[0]["ts"], hits[a.nth - 1]["ts"], w[-1]["ts"]
            pa.append(tm - t0)
            pb.append(t1 - tm)
print(f"waves {len(pa)}: phase A median {st.median(pa):.0f} cycles, phase B median {st.median(pb):.0f} cycles, "
      f"A share {sum(pa) / (sum(pa) + sum(pb)):.1%}")
