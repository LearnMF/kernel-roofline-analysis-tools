"""Per-kernel device time of ours (hip_kda G2) from the round tracker's hipprof traces, keyed by call
occurrence inside one fwd+bwd window ("name#k" in launch order) -- the BW counterpart of
nv_kernel_breakdown.py, input to stage_compare_vs.py.  Median over all windows of all processes.
    python3 bw_kernel_breakdown.py <dev dir with xpd_g2_T_H_r*.json> > bw_kb.json"""
import collections
import glob
import json
import re
import statistics as st
import sys

from kra.timeline.analyze import split_windows
from kra.timeline.hipprof_json import load

per = collections.defaultdict(lambda: collections.defaultdict(list))
for f in sorted(glob.glob(f"{sys.argv[1]}/xpd_g2_*_r*.json")):
    m = re.search(r"xpd_g2_(\d+)_(\d+)_r(\d+)", f)
    key = f"g2_{m.group(1)}_{m.group(2)}"
    for w in split_windows(load(f), "spin_kernel", None):
        seen = collections.Counter()
        for o in w:
            if o.kind != "kernel":
                continue
            mm = re.search(r"(g2_\w+?_kernel|p3_wy_\w+?_kernel)", o.name)
            base = mm.group(1) if mm else re.sub(r"\(.*", "", o.name)[:60]
            seen[base] += 1
            per[key][f"{base}#{seen[base]}"].append(o.dur_ns / 1e3)
print(json.dumps({k: {n: round(st.median(v), 2) for n, v in d.items()} for k, d in per.items()}))
