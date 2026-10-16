"""Is the gap between the F+B wall clock and the kernel-time sum real (GPU idle while the host launches) or an artifact of
synchronizing every repetition?  Times one F+B per repetition with a sync before it (the A/B harness's mode) and N
back-to-back F+Bs with one sync at the end (the host runs ahead, as in training), plus the same with torch.profiler's
kernel list to get the GPU-busy sum.
    PYTHONPATH=<env>:<tree>:<takeover> python3 gap_probe.py --tree TREE --T 16384 --H 12 [--n 10]"""
import argparse
import statistics
import sys

import torch

p = argparse.ArgumentParser()
p.add_argument("--tree", required=True)
p.add_argument("--T", type=int, default=16384)
p.add_argument("--H", type=int, default=12)
p.add_argument("--n", type=int, default=10)
a = p.parse_args()
sys.argv = ["x", str(a.T), str(a.H), "g2", "1"]
src = open(f"{a.tree}/tests/megatron/kda_op_bench_tp.py").read().split("def fwd(arm")[0]
g = {}
exec(compile(src, "kda_op_bench_tp", "exec"), g)
L, D_O, CU, CU_CPU = g["LEAVES"], g["D_O"], g["CU"], g["CU_CPU"]
from hip_kda.fla import chunk_kda  # noqa: E402

kw = dict(scale=None, output_final_state=False, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
          use_beta_sigmoid_in_kernel=False, safe_gate=True, lower_bound=-5.0, cu_seqlens=CU, cu_seqlens_cpu=CU_CPU)


def fb():
    for t_ in L.values():
        t_.grad = None
    out = chunk_kda(L["q"], L["k"], L["v"], L["g"], L["beta"], A_log=L["A_log"], dt_bias=L["dt_bias"], **kw)
    (out[0] if isinstance(out, (tuple, list)) else out).backward(D_O)


for _ in range(3):
    fb()
torch.cuda.synchronize()
single = []
for _ in range(a.n):
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    fb()
    e1.record()
    torch.cuda.synchronize()
    single.append(e0.elapsed_time(e1) * 1e3)
torch.cuda.synchronize()
e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
e0.record()
for _ in range(a.n):
    fb()
e1.record()
torch.cuda.synchronize()
burst = e0.elapsed_time(e1) * 1e3 / a.n
busy = None
try:
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(a.n):
            fb()
        torch.cuda.synchronize()
    busy = sum(e.self_device_time_total if hasattr(e, "self_device_time_total") else e.self_cuda_time_total
               for e in prof.key_averages()) / a.n
except Exception as ex:  # noqa: BLE001
    print("profiler unavailable:", ex)
print(f"T={a.T} H={a.H}: synced single F+B {statistics.median(single):.0f} us | back-to-back {burst:.0f} us/iter"
      + (f" | GPU kernel-busy {busy:.0f} us/iter" if busy else ""))
