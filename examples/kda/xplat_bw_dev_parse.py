"""Parse xplat_bw_dev.sh traces: per (shape, arm) the median over processes of the per-process
median fwd+bwd device time (sum of kernel durations in each spin-marker window).
    python3 xplat_bw_dev_parse.py /public/home/tanbo/xplat/dev > bw_dev.json"""
import collections
import glob
import json
import re
import statistics as st
import sys

from kra.timeline.analyze import split_windows
from kra.timeline.hipprof_json import load

res = collections.defaultdict(list)
for f in sorted(glob.glob(f"{sys.argv[1]}/xpd_*.json")):
    m = re.search(r"xpd_(fla_nr|fla|g2)_(\d+)_(\d+)_r(\d+)", f)
    arm, T, H = m.group(1), int(m.group(2)), int(m.group(3))
    wins = split_windows(load(f), "spin_kernel", None)
    sums = [sum(o.dur_ns for o in w if o.kind == "kernel") / 1e6 for w in wins]
    if sums:
        res[(T, H, arm)].append(st.median(sums))
out = collections.defaultdict(dict)
for (T, H, arm), v in res.items():
    out[f"{T}x{H}"][arm] = {"device_fb_ms": round(st.median(v), 3), "procs": len(v),
                           "spread": round((max(v) - min(v)) / st.median(v), 4)}
print(json.dumps(out, indent=1, sort_keys=True))
