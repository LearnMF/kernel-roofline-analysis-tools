import os, sys, collections, re, torch
sys.argv = ["x", sys.argv[1], sys.argv[2], "g2,ftc", "1"]
src = open("/public/home/tanbo/xplat/kda_op_bench_ftgh.py").read().split("res, live = {}, []")[0]
g = {}; exec(compile(src, "b", "exec"), g)
from torch.profiler import profile, ProfilerActivity
res = {}
for arm in os.environ["ARMS"].split(","):
    for _ in range(2): o = g["fwd"](arm); g["bwd"](o); del o
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(3): o = g["fwd"](arm); g["bwd"](o); del o
        torch.cuda.synchronize()
    c = collections.Counter()
    for e in p.key_averages():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            k = re.sub(r"\(.*", "", e.key); k = re.sub(r"^void ", "", k)[:48]; c[k] += e.device_time_total / 3000
    res[arm] = c
a, b = res["g2"], res["ftc"]
print(f"total g2 {sum(a.values()):.3f} ftc {sum(b.values()):.3f}")
for k in sorted(set(a) | set(b), key=lambda k: -(a[k] + b[k]))[:16]:
    print(f"  {k:48s} {a[k]:8.3f} {b[k]:8.3f} {b[k]/a[k] if a[k] else 0:6.3f}")
