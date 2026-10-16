"""Replay one recorded production g2_ops call with keyword-argument variants (device-event timing, interleaved),
plus an optional bitwise check of a variant's output against another recorded call's output.
    PYTHONPATH=<env>:<tree>:<takeover> python3 op_variant_time.py --tree TREE --fn fwd_h_o --T 16384 --H 12 \
        --variant base: --variant bn:emit_bn=1,emit_vn=0 [--check bn:2=fwd_h_bn:0]
--check V:i=F:j compares output i of variant V with output j of the recorded call F (bitwise)."""
import argparse
import statistics
import sys

import torch

p = argparse.ArgumentParser()
p.add_argument("--tree", required=True)
p.add_argument("--fn", required=True)
p.add_argument("--T", type=int, default=16384)
p.add_argument("--H", type=int, default=12)
p.add_argument("--variant", action="append", required=True)
p.add_argument("--check", action="append", default=[])
p.add_argument("--reps", type=int, default=15)
a = p.parse_args()
sys.argv = ["x", str(a.T), str(a.H), "g2", "1"]
src = open(f"{a.tree}/tests/megatron/kda_op_bench_tp.py").read().split("def fwd(arm")[0]
g = {}
exec(compile(src, "kda_op_bench_tp", "exec"), g)
L, D_O, CU, CU_CPU = g["LEAVES"], g["D_O"], g["CU"], g["CU_CPU"]
from hip_kda.fla import chunk_kda  # noqa: E402
from hip_kda.ops import g2_ops  # noqa: E402

names = {a.fn} | {c.split("=")[1].split(":")[0] for c in a.check}
REC, OUT = {}, {}
for nm in names:
    f0 = getattr(g2_ops, nm)

    def mk(f0, nm):
        def w(*args, **kw):
            o = f0(*args, **kw)
            if nm not in REC:
                REC[nm] = (args, dict(kw), f0)
                OUT[nm] = [t.clone() if isinstance(t, torch.Tensor) and t.numel() else t
                           for t in (o if isinstance(o, (tuple, list)) else [o])]
            return o
        return w
    setattr(g2_ops, nm, mk(f0, nm))
kw = dict(scale=None, output_final_state=False, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
          use_beta_sigmoid_in_kernel=False, safe_gate=True, lower_bound=-5.0, cu_seqlens=CU, cu_seqlens_cpu=CU_CPU)
out = chunk_kda(L["q"], L["k"], L["v"], L["g"], L["beta"], A_log=L["A_log"], dt_bias=L["dt_bias"], **kw)
(out[0] if isinstance(out, (tuple, list)) else out).backward(D_O)
torch.cuda.synchronize()
args, kws, f = REC[a.fn]


def parse(v):
    name, _, kv = v.partition(":")
    d = dict(kws)
    for item in filter(None, kv.split(",")):
        k, _, val = item.partition("=")
        d[k] = bool(int(val)) if val in ("0", "1") else val
    return name, d


variants = [parse(v) for v in a.variant]
res = {n: [] for n, _ in variants}
last = {}
for r in range(a.reps + 3):
    for n, d in variants:
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        o = f(*args, **d)
        e1.record()
        torch.cuda.synchronize()
        if r == 0:
            last[n] = list(o) if isinstance(o, (tuple, list)) else [o]
        if r >= 3:
            res[n].append(e0.elapsed_time(e1) * 1e3)


def bits(t):
    return t.view(torch.int16) if t.dtype in (torch.bfloat16, torch.float16) else t


for c in a.check:
    lhs, rhs = c.split("=")
    vn, vi = lhs.split(":")
    fn, fi = rhs.split(":")
    x, y = last[vn][int(vi)], OUT[fn][int(fi)]
    same = x.shape == y.shape and torch.equal(bits(x.contiguous()), bits(y.contiguous()))
    print(f"check {c}: {'BITWISE' if same else 'DIFF'} {tuple(x.shape)} vs {tuple(y.shape)}")
base = statistics.median(res[variants[0][0]])
for n, _ in variants:
    m = statistics.median(res[n])
    print(f"{a.fn} {n:>8s}: {m:9.1f} us  ({m - base:+8.1f} us vs {variants[0][0]})  spread {100 * (max(res[n]) - min(res[n])) / m:.1f}%")
