"""Analyse issue_burst_bench SQTT traces: per burst, the issue time (relative to the burst's first instruction) of
every burst instruction, averaged over waves and over rounds 2..R (round 1 still warms up).
Reports the index of the first instruction whose issue gap exceeds 3x the median early gap (= the in-flight limit)
and the early issue interval.
    python3 burst_sqtt.py <dir> <opcode-prefix>"""
import collections
import glob
import json
import sys

d, pref = sys.argv[1], sys.argv[2]
prof = collections.defaultdict(list)
nw = 0
for f in sorted(glob.glob(f"{d}/*thread_trace_*__*.json")):
    ev = json.load(open(f))["traceEvents"]
    st = collections.defaultdict(list)
    for e in ev:
        if e.get("ph") == "X" and "assembly" in e.get("args", {}):
            st[(e["pid"], e["tid"])].append(e)
    for s in st.values():
        s.sort(key=lambda e: (e["ts"], e["args"].get("inst idx", 0)))
        bursts, cur = [], []
        for e in s:
            if e["args"]["assembly"].startswith(pref):
                cur.append(e["ts"])
            elif e["name"] == "s_waitcnt" and cur:
                bursts.append(cur)
                cur = []
        if len(bursts) < 3:
            continue
        nw += 1
        for b in bursts[1:]:
            for i, t in enumerate(b):
                prof[i].append(t - b[0])
n = len(prof)
avg = [sum(prof[i]) / len(prof[i]) for i in range(n)]
gaps = [avg[i] - avg[i - 1] for i in range(1, n)]
early = sorted(gaps[:3])[1] if len(gaps) >= 3 else (gaps[0] if gaps else 0)
lim = next((i for i, g in enumerate(gaps, start=1) if g > max(3 * early, early + 40)), None)
print(f"{d}: {nw} waves, burst length {n}; early issue interval {early:.1f} cyc; "
      f"first blocking at instruction #{lim if lim is not None else '-'} "
      f"(gap {gaps[lim - 1]:.0f} cyc)" if lim else f"{d}: {nw} waves, burst length {n}; issue interval {early:.1f}; no blocking")
print("  issue time per instruction:", " ".join(f"{a:.0f}" for a in avg))
