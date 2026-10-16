"""Replay one recorded production g2_ops call (fwd_h_bn / dhu_bn / ...) with an env switch toggled per repetition:
device-event time per value + bitwise check of every output against value[0].
    PYTHONPATH=<env>:<tree>:<takeover> python3 op_env_time.py --tree TREE --fn fwd_h_bn --T 16384 --H 12 \
        --env KDA_G2_REC_UV --vals 0,1,2"""
import argparse
import os
import statistics
import sys

import torch

p = argparse.ArgumentParser()
p.add_argument("--tree", required=True)
p.add_argument("--fn", default="fwd_h_bn")
p.add_argument("--T", type=int, default=16384)
p.add_argument("--H", type=int, default=12)
p.add_argument("--env", required=True)
p.add_argument("--vals", default="0,1")
p.add_argument("--reps", type=int, default=20)
a = p.parse_args()
sys.argv = ["x", str(a.T), str(a.H), "g2", "1"]
src = open(f"{a.tree}/tests/megatron/kda_op_bench_tp.py").read().split("def fwd(arm")[0]
g = {}
exec(compile(src, "kda_op_bench_tp", "exec"), g)
L, D_O, CU, CU_CPU = g["LEAVES"], g["D_O"], g["CU"], g["CU_CPU"]
from hip_kda.fla import chunk_kda  # noqa: E402
from hip_kda.ops import g2_ops  # noqa: E402

REC = {}
fn0 = getattr(g2_ops, a.fn)


def wrap(*args, **kw):
    REC.setdefault("c", (args, kw))
    return fn0(*args, **kw)


setattr(g2_ops, a.fn, wrap)
kw = dict(scale=None, output_final_state=False, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
          use_beta_sigmoid_in_kernel=False, safe_gate=True, lower_bound=-5.0, cu_seqlens=CU, cu_seqlens_cpu=CU_CPU)
out = chunk_kda(L["q"], L["k"], L["v"], L["g"], L["beta"], A_log=L["A_log"], dt_bias=L["dt_bias"], **kw)
(out[0] if isinstance(out, (tuple, list)) else out).backward(D_O)
torch.cuda.synchronize()
args, kws = REC["c"]
vals = a.vals.split(",")


def flat(o):
    o = o if isinstance(o, (tuple, list)) else [o]
    return [t for t in o if isinstance(t, torch.Tensor) and t.numel()]


def bits(t):
    return t.view(torch.int16) if t.dtype in (torch.bfloat16, torch.float16) else t


ref = None
res = {v: [] for v in vals}
for r in range(a.reps + 3):
    for v in vals:
        os.environ[a.env] = v
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        o = fn0(*args, **kws)
        e1.record()
        torch.cuda.synchronize()
        if r == 0:
            outs = flat(o)
            if ref is None:
                ref = [t.clone() for t in outs]
            else:
                same = len(outs) == len(ref) and all(torch.equal(bits(x), bits(y)) for x, y in zip(outs, ref))
                print(f"{a.env}={v}: outputs {'BITWISE' if same else 'DIFF'} vs {vals[0]}")
        if r >= 3:
            res[v].append(e0.elapsed_time(e1) * 1e3)
base = statistics.median(res[vals[0]])
for v in vals:
    m = statistics.median(res[v])
    print(f"{a.fn} {a.env}={v}: {m:8.1f} us  ({base / m:.3f}x vs {vals[0]})  spread {100 * (max(res[v]) - min(res[v])) / m:.1f}%")
