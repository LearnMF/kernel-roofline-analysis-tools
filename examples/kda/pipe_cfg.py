"""Wall-clock A/B of the g2 forward and forward+backward over several env CONFIGS toggled per repetition in one process.
    PYTHONPATH=<env>:<tree>:<takeover> python3 pipe_cfg.py --tree TREE --T 16384 --H 48 \
        --cfg base: --cfg kp:KDA_G2_KPERM_MAX_H=96 --cfg kp_h:KDA_G2_KPERM_MAX_H=96,KDA_G2_SAVE_H_MAX_H=96 [--reps 12]
The first config is the reference; every config also reports a bitwise check of o and the gradients vs it."""
import argparse
import os
import statistics
import sys

import torch

p = argparse.ArgumentParser()
p.add_argument("--tree", required=True)
p.add_argument("--T", type=int, default=16384)
p.add_argument("--H", type=int, default=12)
p.add_argument("--cfg", action="append", required=True)
p.add_argument("--reps", type=int, default=12)
a = p.parse_args()
sys.argv = ["x", str(a.T), str(a.H), "g2", "1"]
src = open(f"{a.tree}/tests/megatron/kda_op_bench_tp.py").read().split("def fwd(arm")[0]
g = {}
exec(compile(src, "kda_op_bench_tp", "exec"), g)
L, D_O, CU, CU_CPU = g["LEAVES"], g["D_O"], g["CU"], g["CU_CPU"]
from hip_kda.fla import chunk_kda  # noqa: E402

kw = dict(scale=None, output_final_state=False, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
          use_beta_sigmoid_in_kernel=False, safe_gate=True, lower_bound=-5.0, cu_seqlens=CU, cu_seqlens_cpu=CU_CPU)
cfgs = []
keys = set()
for c in a.cfg:
    name, _, kv = c.partition(":")
    d = dict(x.split("=", 1) for x in kv.split(",") if x)
    keys |= set(d)
    cfgs.append((name, d))
saved = {k: os.environ.get(k) for k in keys}


def setenv(d):
    for k in keys:
        if k in d:
            os.environ[k] = d[k]
        elif saved[k] is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = saved[k]


def ev():
    return torch.cuda.Event(enable_timing=True)


def bits(t):
    return t.view(torch.int16) if t.dtype in (torch.bfloat16, torch.float16) else t


res = {(n, w): [] for n, _ in cfgs for w in ("f", "fb")}
ref = None
for r in range(a.reps + 3):
    for n, d in cfgs:
        setenv(d)
        for t_ in L.values():
            t_.grad = None
        torch.cuda.synchronize()
        e0, e1, e2 = ev(), ev(), ev()
        e0.record()
        out = chunk_kda(L["q"], L["k"], L["v"], L["g"], L["beta"], A_log=L["A_log"], dt_bias=L["dt_bias"], **kw)
        o = out[0] if isinstance(out, (tuple, list)) else out
        e1.record()
        o.backward(D_O)
        e2.record()
        torch.cuda.synchronize()
        if r == 0:
            outs = [o.detach().clone()] + [t_.grad.clone() for t_ in L.values() if t_.grad is not None]
            if ref is None:
                ref = outs
            else:
                same = all(torch.equal(bits(x), bits(y)) for x, y in zip(outs, ref))
                print(f"cfg {n}: outputs {'BITWISE' if same else 'DIFF'} vs {cfgs[0][0]}")
        if r >= 3:
            res[(n, "f")].append(e0.elapsed_time(e1) * 1e3)
            res[(n, "fb")].append(e0.elapsed_time(e2) * 1e3)
b_f = statistics.median(res[(cfgs[0][0], "f")])
b_fb = statistics.median(res[(cfgs[0][0], "fb")])
for n, _ in cfgs:
    mf, mfb = statistics.median(res[(n, "f")]), statistics.median(res[(n, "fb")])
    print(f"T={a.T} H={a.H} {n:>8s}: fwd {mf:7.0f} us ({b_f / mf:.3f}x)   fwd+bwd {mfb:7.0f} us ({b_fb / mfb:.3f}x)")
