"""Wall-clock A/B of the g2 forward (and forward+backward) with an env switch toggled per repetition in one process
(kernel-sum A/B cannot see cross-stream overlap).
    PYTHONPATH=<env>:<tree>:<takeover> python3 pipe_time.py --tree TREE --T 16384 --H 12 --env KDA_G2_PIPE [--reps 30]"""
import argparse
import os
import statistics
import sys

import torch

p = argparse.ArgumentParser()
p.add_argument("--tree", required=True)
p.add_argument("--T", type=int, default=16384)
p.add_argument("--H", type=int, default=12)
p.add_argument("--env", default="KDA_G2_PIPE")
p.add_argument("--reps", type=int, default=30)
a = p.parse_args()
sys.argv = ["x", str(a.T), str(a.H), "g2", "1"]
src = open(f"{a.tree}/tests/megatron/kda_op_bench_tp.py").read().split("def fwd(arm")[0]
g = {}
exec(compile(src, "kda_op_bench_tp", "exec"), g)
L, D_O, CU, CU_CPU = g["LEAVES"], g["D_O"], g["CU"], g["CU_CPU"]
from hip_kda.fla import chunk_kda  # noqa: E402

kw = dict(scale=None, output_final_state=False, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
          use_beta_sigmoid_in_kernel=False, safe_gate=True, lower_bound=-5.0, cu_seqlens=CU, cu_seqlens_cpu=CU_CPU)


def fwd():
    out = chunk_kda(L["q"], L["k"], L["v"], L["g"], L["beta"], A_log=L["A_log"], dt_bias=L["dt_bias"], **kw)
    return out[0] if isinstance(out, (tuple, list)) else out


def ev():
    return torch.cuda.Event(enable_timing=True)


res = {("0", "f"): [], ("1", "f"): [], ("0", "fb"): [], ("1", "fb"): []}
for r in range(a.reps + 3):
    for flag in ("0", "1"):
        os.environ[a.env] = flag
        for t_ in L.values():
            t_.grad = None
        torch.cuda.synchronize()
        e0, e1, e2 = ev(), ev(), ev()
        e0.record()
        o = fwd()
        e1.record()
        o.backward(D_O)
        e2.record()
        torch.cuda.synchronize()
        if r >= 3:
            res[(flag, "f")].append(e0.elapsed_time(e1) * 1e3)
            res[(flag, "fb")].append(e0.elapsed_time(e2) * 1e3)
m = {k_: statistics.median(v_) for k_, v_ in res.items()}
print(f"T={a.T} H={a.H} {a.env}: fwd {m[('0', 'f')]:.0f} -> {m[('1', 'f')]:.0f} us "
      f"({m[('0', 'f')] / m[('1', 'f')]:.3f}x)   fwd+bwd {m[('0', 'fb')]:.0f} -> {m[('1', 'fb')]:.0f} us "
      f"({m[('0', 'fb')] / m[('1', 'fb')]:.3f}x)")
