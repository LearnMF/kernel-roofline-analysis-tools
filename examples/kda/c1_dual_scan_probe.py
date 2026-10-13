"""C1 upper bound (cuDNN FE Hopper KDA backward's dual-direction k_scan): the backward's two recurrences
fwd_h_bn (h / v_new recompute) and dhu_bn (dh) do not depend on each other once dav is split (dhu reads
dOT / dVn = tril(Aqk)^T do, which never touch Vn).  Their arguments are recorded from a real fwd+bwd and
replayed (same operand set: inputs are read-only, outputs disjoint -- the real schedule's situation):
  serial   : fwd_h_bn ; dhu_bn                       (today)
  together : fwd_h_bn on stream A || dhu_bn on stream B
Reports per-kernel alone times, the serial sum, the concurrent wall time and the saving.  Contention only:
the dav split cost (one more launch, do re-read) is charged separately.
    python3 c1_dual_scan_probe.py --tree <g2 tree> --T 16384 --H 12"""
import argparse
import statistics as st
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--tree", required=True)
ap.add_argument("--T", type=int, default=16384)
ap.add_argument("--H", type=int, default=12)
ap.add_argument("--n", type=int, default=15)
ap.add_argument("--rounds", type=int, default=3)
a = ap.parse_args()
sys.argv = ["x", str(a.T), str(a.H), "g2", "1"]
src = open(f"{a.tree}/tests/megatron/kda_op_bench_tp.py").read().split("res, live = {}, []")[0]
g = {}
exec(compile(src, "kda_op_bench_tp", "exec"), g)

import hip_kda.ops.g2_ops as g2_ops  # noqa: E402

NAMES = ("fwd_h_bn", "dhu_bn")
rec = {}
orig = {n: getattr(g2_ops, n) for n in NAMES}


def wrap(n):
    def w(*args, **kw):
        rec.setdefault(n, (args, kw))
        return orig[n](*args, **kw)
    return w


for n in NAMES:
    setattr(g2_ops, n, wrap(n))
o = g["fwd"]("g2")
g["bwd"](o)
torch.cuda.synchronize()
for n in NAMES:
    setattr(g2_ops, n, orig[n])
assert all(n in rec for n in NAMES), list(rec)


def call(n):
    args, kw = rec[n]
    return orig[n](*args, **kw)


sa, sb = torch.cuda.Stream(), torch.cuda.Stream()


def timed(fn):
    ts = []
    for it in range(a.n + 3):
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record(); torch.cuda.synchronize()
        if it >= 3:
            ts.append(e0.elapsed_time(e1))
    return st.median(ts)


def together():
    ra, rb, rw = [], [], []
    for it in range(a.n + 3):
        torch.cuda.synchronize()
        c = torch.cuda.current_stream()
        e0 = torch.cuda.Event(enable_timing=True); e0.record(c)
        sa.wait_stream(c); sb.wait_stream(c)
        ea, eb = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(sa):
            call("fwd_h_bn"); ea.record(sa)
        with torch.cuda.stream(sb):
            call("dhu_bn"); eb.record(sb)
        torch.cuda.synchronize()
        if it >= 3:
            ta, tb = e0.elapsed_time(ea), e0.elapsed_time(eb)
            ra.append(ta); rb.append(tb); rw.append(max(ta, tb))
    return st.median(ra), st.median(rb), st.median(rw)


for r in range(a.rounds):
    tf, td = timed(lambda: call("fwd_h_bn")), timed(lambda: call("dhu_bn"))
    ts = timed(lambda: (call("fwd_h_bn"), call("dhu_bn")))
    ca, cb, cw = together()
    print(f"T={a.T} H={a.H} round {r}: fwd_h_bn {tf * 1e3:.0f} us, dhu {td * 1e3:.0f} us, serial {ts * 1e3:.0f} us | "
          f"together: fwd_h_bn {ca * 1e3:.0f} us, dhu {cb * 1e3:.0f} us, wall {cw * 1e3:.0f} us "
          f"-> saving {(ts - cw) * 1e3:.0f} us ({100 * (ts - cw) / ts:.1f}% of serial)", flush=True)
