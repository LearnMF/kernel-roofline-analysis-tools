import os, sys, collections, re, torch
sys.argv = ["x", sys.argv[1], sys.argv[2], "g2,ftc", "1"]
src = open("/public/home/tanbo/xplat/kda_op_bench_ftgh.py").read().split("res, live = {}, []")[0]
g = {}; exec(compile(src, "b", "exec"), g)
from torch.profiler import profile, ProfilerActivity
for arm in os.environ["ARMS"].split(","):
    for _ in range(2): o = g["fwd"](arm); g["bwd"](o); del o
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(3): o = g["fwd"](arm); g["bwd"](o); del o
        torch.cuda.synchronize()
    c = collections.Counter()
    for e in p.key_averages():
        if e.device_type == torch.autograd.DeviceType.CUDA: c[re.sub(r"\(.*", "", e.key)[:40]] += e.device_time_total / 3000
    tot = sum(c.values())
    print(arm, os.environ.get("BENCH_DENSE"), f"total {tot:.3f}", " ".join(f"{k.split()[-1]}={v:.3f}" for k, v in c.most_common(5)))
