"""flash_train.torch.kimi_delta_attention wall-clock A/B over env CONFIGS toggled per repetition in one process
(pipe_cfg.py for flash-train), with a bitwise check of o + grads vs the first config.
    PYTHONPATH=<ft_site>:<env>:<tree>:<takeover> python3 ft_pipe_cfg.py --tree TREE --T 16384 --H 12 \
        --cfg ks_old:FLASH_TRAIN_KDA_KEEP_STATE=1,KDA_G2_SAVE_H=0 --cfg ks_h:FLASH_TRAIN_KDA_KEEP_STATE=1 [--reps 8]"""
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
p.add_argument("--reps", type=int, default=8)
a = p.parse_args()
sys.argv = ["x", str(a.T), str(a.H), "ft", "1"]
src = open(f"{a.tree}/tests/megatron/kda_op_bench_tp.py").read().split("def fwd(arm")[0]
g = {}
exec(compile(src, "kda_op_bench_tp", "exec"), g)
L, D_O, CU = g["LEAVES"], g["D_O"], g["CU"]
import flash_train.torch as ft  # noqa: E402

cfgs, keys = [], set()
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
        e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        e0.record()
        o = ft.kimi_delta_attention(L["q"][0], L["k"][0], L["v"][0], L["g"][0], L["beta"][0], A_log=L["A_log"],
                                    dt_bias=L["dt_bias"], cu_seqlens=CU).unsqueeze(0)
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
b_f, b_fb = (statistics.median(res[(cfgs[0][0], w)]) for w in ("f", "fb"))
for n, _ in cfgs:
    mf, mfb = statistics.median(res[(n, "f")]), statistics.median(res[(n, "fb")])
    print(f"FT T={a.T} H={a.H} {n:>8s}: fwd {mf:7.0f} us ({b_f / mf:.3f}x)   fwd+bwd {mfb:7.0f} us ({b_fb / mfb:.3f}x)")
