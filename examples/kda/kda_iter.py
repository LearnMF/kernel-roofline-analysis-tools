"""KDA forward+backward driver for timeline capture (L1).

Runs `iters` measured fwd+bwd iterations of one arm of the Kimi K3 KDA op bench,
each preceded by a marker kernel (torch.cuda._sleep -> a kernel whose name contains
"sleep") so `kra timeline --marker sleep` can cut the trace into per-iteration windows.

Usage (inside the KDA container, under hipprof):
    hipprof --hip-trace --output-type 0 -o OUT \
        python3 kda_iter.py --tree /opt/kda_env/g2_r5 --arm g2 --T 8192 --H 12 --iters 5

`--sync` inserts a device synchronize between fwd and bwd (default: no sync, like training).
`--time` (run WITHOUT hipprof) prints the unprofiled per-iteration time measured with
events around the same fwd+bwd window -- the reference for `kra timeline --reference-ms`.
"""
import argparse
import sys

import torch

p = argparse.ArgumentParser()
p.add_argument("--tree", default="/opt/kda_env/g2_r5")
p.add_argument("--arm", default="g2", help="g2 | fla | fla_nr")
p.add_argument("--T", type=int, default=8192)
p.add_argument("--H", type=int, default=12)
p.add_argument("--iters", type=int, default=5)
p.add_argument("--warmup", type=int, default=3)
p.add_argument("--sync", action="store_true")
p.add_argument("--time", action="store_true", help="print unprofiled per-iteration ms")
a = p.parse_args()

# kda_op_bench_tp.py defines the inputs and fwd()/bwd() closures for each arm before
# its own timing loop starts at "res, live = {}, []".
sys.argv = ["x", str(a.T), str(a.H), a.arm, "1"]
src = open(f"{a.tree}/tests/megatron/kda_op_bench_tp.py").read().split("res, live = {}, []")[0]
g = {}
exec(compile(src, "kda_op_bench_tp", "exec"), g)

for _ in range(a.warmup):
    o = g["fwd"](a.arm)
    g["bwd"](o)
    del o
torch.cuda.synchronize()
times = []
for _ in range(a.iters):
    torch.cuda._sleep(200000)          # marker kernel; analysis windows start after it
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    o = g["fwd"](a.arm)
    if a.sync:
        torch.cuda.synchronize()
    g["bwd"](o)
    e1.record()
    del o
    torch.cuda.synchronize()
    times.append(e0.elapsed_time(e1))
if a.time:
    s = sorted(times)
    print(f"KDA_ITER_MS median={s[len(s) // 2]:.4f} min={s[0]:.4f} max={s[-1]:.4f} n={len(s)}")
print("KDA_ITER_DONE", a.arm, a.T, a.H, a.iters)
