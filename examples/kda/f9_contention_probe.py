"""F9 input: co-residency contention coefficients of the BACKWARD chain kernels and their parallel
partners on gfx936 (the simulator f9_sched_sim.py borrowed the forward's P-41 values 0.48 / 0.71).

The backward launchers (g2_ops.wu, fwd_h_bn, dav, dhu_bn, wy_s4_split) are wrapped to record their
arguments during two real fwd+bwd runs (two independent operand sets), then replayed:
  pair 1: chain fwd_h_bn (set 1)  ||  parallel wu + dav (set 2)
  pair 2: chain dhu_bn   (set 1)  ||  parallel wy_s4_split (s4a + s4b + tail, set 2)
alone and concurrently on two streams with per-stream events.  From the overlap window
  s = chain speed while co-resident, p = parallel speed while co-resident
(the part of the longer stream that runs after the other finished is credited at speed 1).
This measures contention only; dependency effects are the simulator's job (R35).
    python3 f9_contention_probe.py --tree <g2 tree> --T 16384 --H 12"""
import argparse
import statistics as st
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--tree", required=True)
ap.add_argument("--T", type=int, default=16384)
ap.add_argument("--H", type=int, default=12)
ap.add_argument("--n", type=int, default=9)
a = ap.parse_args()
sys.argv = ["x", str(a.T), str(a.H), "g2", "1"]
src = open(f"{a.tree}/tests/megatron/kda_op_bench_tp.py").read().split("res, live = {}, []")[0]
g = {}
exec(compile(src, "kda_op_bench_tp", "exec"), g)

import hip_kda.ops.g2_ops as g2_ops  # noqa: E402

NAMES = ("wu", "fwd_h_bn", "dav", "dhu_bn", "wy_s4_split")
rec = [dict(), dict()]
cur = [0]
orig = {n: getattr(g2_ops, n) for n in NAMES}


def wrap(n):
    def w(*args, **kw):
        rec[cur[0]].setdefault(n, (args, kw))
        return orig[n](*args, **kw)
    return w


for n in NAMES:
    setattr(g2_ops, n, wrap(n))
for s in (0, 1):
    cur[0] = s
    o = g["fwd"]("g2")
    g["bwd"](o)
    del o
torch.cuda.synchronize()
for n in NAMES:
    setattr(g2_ops, n, orig[n])
assert all(n in rec[0] and n in rec[1] for n in NAMES), [list(r) for r in rec]


def call(s, n):
    args, kw = rec[s][n]
    return orig[n](*args, **kw)


sa, sb = torch.cuda.Stream(), torch.cuda.Stream()


def alone(fn):
    ts = []
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    for _ in range(a.n):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return st.median(ts)


def together(fa, fb):
    ra, rb = [], []
    for it in range(a.n + 3):
        torch.cuda.synchronize()
        c = torch.cuda.current_stream()
        e0 = torch.cuda.Event(enable_timing=True); e0.record(c)
        sa.wait_stream(c); sb.wait_stream(c)
        ea, eb = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(sa):
            fa(); ea.record(sa)
        with torch.cuda.stream(sb):
            fb(); eb.record(sb)
        torch.cuda.synchronize()
        if it >= 3:
            ra.append(e0.elapsed_time(ea)); rb.append(e0.elapsed_time(eb))
    return st.median(ra), st.median(rb)


def coeffs(name, chain, par):
    tc, tp = alone(chain), alone(par)
    cc, cp = together(chain, par)
    if cc <= cp:                                  # chain finished first: par ran alone afterwards
        s = tc / cc
        p = (tp - (cp - cc)) / cc
    else:
        p = tp / cp
        s = (tc - (cc - cp)) / cp
    print(f"{name}: chain alone {tc:.3f} ms, parallel alone {tp:.3f} ms, serial {tc + tp:.3f} ms | "
          f"together: chain {cc:.3f} ms, parallel {cp:.3f} ms, wall {max(cc, cp):.3f} ms "
          f"-> s_chain {s:.2f}, s_par {p:.2f}")


coeffs("pair1 fwd_h_bn || wu+dav", lambda: call(0, "fwd_h_bn"), lambda: (call(1, "wu"), call(1, "dav")))
coeffs("pair2 dhu_bn || s4a+s4b+tail", lambda: call(0, "dhu_bn"), lambda: call(1, "wy_s4_split"))
