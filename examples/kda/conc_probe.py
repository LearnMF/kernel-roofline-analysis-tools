"""Concurrency feasibility probe (C-1): can the chunk-parallel producers run on the CUs the TP8 recurrence leaves
idle (48 of 80 CTAs at H = 12)?  Times, on the K3 bench inputs:
  prep      g2 fwd_prep (prep_a + prep_b, chunk-parallel)
  fwdh      fwd_h_o (forward recurrence, 4*H CTAs) on precomputed operands
  both      the two launched together on two streams (independent operand sets)
and reports overlap = (prep + fwdh - both) / min(prep, fwdh).  An upper bound for a chunk-pipelined
producer/recurrence schedule: if two independent launches cannot overlap, a pipelined one cannot either.
    PYTHONPATH=<env>:<tree>:<takeover> python3 conc_probe.py --tree TREE --T 16384 --H 12 [--reps 20]"""
import argparse
import statistics
import sys

import torch

p = argparse.ArgumentParser()
p.add_argument("--tree", required=True)
p.add_argument("--T", type=int, default=16384)
p.add_argument("--H", type=int, default=12)
p.add_argument("--reps", type=int, default=20)
a = p.parse_args()
sys.argv = ["x", str(a.T), str(a.H), "g2", "1"]
src = open(f"{a.tree}/tests/megatron/kda_op_bench_tp.py").read().split("def fwd(arm")[0]
g = {}
exec(compile(src, "kda_op_bench_tp", "exec"), g)
X = g["X"]
from hip_kda.ops import g2_ops  # noqa: E402

T, H = a.T, a.H
NT = T // 64
RCP_LN2 = 1.4426950216
q, k, v, graw = X["q"].detach(), X["k"].detach(), X["v"].detach(), X["g"].detach()
beta, alog, dtb = X["beta"].detach().float(), X["A_log"].detach().float(), X["dt_bias"].detach().float()
scale = 128 ** -0.5


def prep():
    return g2_ops.fwd_prep(q, k, v, graw, alog, dtb, beta, -5.0, RCP_LN2, True, scale, H, NT)


Wn, Kn, Un, QgA, AqkA, Aqk, Akk, qr, kr, Gn = prep()


def fwdh():
    return g2_ops.fwd_h_o(Wn, Kn, Un, Gn, QgA, AqkA, scale, emit_bn=False, o_bf16=True, emit_vn=False)


s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()


def timed(fn):
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    st.record()
    fn()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) * 1e3


def both(rev=False):
    ev = torch.cuda.Event()
    ev.record()
    s1.wait_event(ev); s2.wait_event(ev)
    order = [(s2, fwdh), (s1, prep)] if rev else [(s1, prep), (s2, fwdh)]
    for st_, fn in order:
        with torch.cuda.stream(st_):
            fn()
    e1, e2 = torch.cuda.Event(), torch.cuda.Event()
    e1.record(s1); e2.record(s2)
    torch.cuda.current_stream().wait_event(e1); torch.cuda.current_stream().wait_event(e2)


for _ in range(3):
    prep(); fwdh(); both(); both(True)
res = {"prep": [], "fwdh": [], "both": [], "both_rev": []}
for _ in range(a.reps):
    res["prep"].append(timed(prep))
    res["fwdh"].append(timed(fwdh))
    res["both"].append(timed(both))
    res["both_rev"].append(timed(lambda: both(True)))
m = {k_: statistics.median(v_) for k_, v_ in res.items()}
for key in ("both", "both_rev"):
    ov = (m["prep"] + m["fwdh"] - m[key]) / min(m["prep"], m["fwdh"])
    print(f"T={T} H={H}: prep {m['prep']:.1f} us  fwdh {m['fwdh']:.1f} us  {key} {m[key]:.1f} us  "
          f"sum {m['prep'] + m['fwdh']:.1f}  overlap {100 * ov:.0f}% of the shorter")
