"""Per-kernel device time of one FLA chunk_kda fwd+bwd on NVIDIA (CUPTI via torch.profiler), the
NV counterpart of the hipprof per-kernel breakdown -- input to the stage-level competitor
comparison.  Kernels are keyed by call occurrence inside one fwd+bwd ("name#k", in launch
order), so the forward call of a kernel and its backward-recompute call stay apart.
    python3 nv_kernel_breakdown.py T H [arm=fla] [iters=5]"""
import collections
import json
import statistics as st
import sys

import torch
from torch.profiler import ProfilerActivity, profile

T, H = int(sys.argv[1]), int(sys.argv[2])
arm = sys.argv[3] if len(sys.argv) > 3 else "fla"
iters = int(sys.argv[4]) if len(sys.argv) > 4 else 5
sys.argv = ["x", str(T), str(H), arm, "1"]
src = open("/xplat/kda_op_bench_nv.py").read().split("res, live = {}, []")[0]
g = {}
exec(compile(src, "kda_op_bench_nv", "exec"), g)
for _ in range(3):
    o = g["fwd"](arm); g["bwd"](o); del o
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for _ in range(iters):
        torch.cuda.synchronize()
        torch.cuda._sleep(1000)           # marker kernel: one window per iteration
        o = g["fwd"](arm); g["bwd"](o); del o
    torch.cuda.synchronize()
ev = sorted((e for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA),
            key=lambda e: e.time_range.start)
wins, cur = [], None
for e in ev:
    if "sleep" in e.name.lower() or "spin_kernel" in e.name:
        cur = []
        wins.append(cur)
    elif cur is not None:
        cur.append(e)
acc = collections.defaultdict(list)
for w in wins:
    seen = collections.Counter()
    per = collections.Counter()
    for e in w:
        n = e.name
        seen[n] += 1
        per[f"{n}#{seen[n]}"] += (e.time_range.end - e.time_range.start) / 1000.0
    for k, v in per.items():
        acc[k].append(v)
med = {k: st.median(v) for k, v in acc.items()}
print(json.dumps({"T": T, "H": H, "arm": arm, "iters": len(wins), "total_ms": round(sum(med.values()), 3),
                  "kernels": sorted(([k, round(v, 4)] for k, v in med.items()), key=lambda x: -x[1])}))
