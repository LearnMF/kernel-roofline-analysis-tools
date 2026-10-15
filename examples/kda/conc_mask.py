"""C-4: contention-only bound for a CU-partitioned overlap.  The forward recurrence runs on a stream masked to RCU CUs,
an independent full prep on a stream masked to the remaining CUs (hipExtStreamCreateWithCUMask via ctypes,
wrapped as torch ExternalStreams); no flags, no producer/consumer dependency.  Reports each kernel's time alone,
on its own partition alone, and concurrently (per-stream events), so the recurrence slowdown caused purely by
sharing L2/HBM -- not by flag waits or stream hand-offs -- is visible.
    PYTHONPATH=<env>:<tree>:<takeover> python3 conc_mask.py --tree TREE --T 16384 --H 12 [--rcu 48]"""
import argparse
import ctypes
import statistics
import sys

import torch

p = argparse.ArgumentParser()
p.add_argument("--tree", required=True)
p.add_argument("--T", type=int, default=16384)
p.add_argument("--H", type=int, default=12)
p.add_argument("--rcu", type=int, default=48)
p.add_argument("--reps", type=int, default=15)
a = p.parse_args()
sys.argv = ["x", str(a.T), str(a.H), "g2", "1"]
src = open(f"{a.tree}/tests/megatron/kda_op_bench_tp.py").read().split("def fwd(arm")[0]
g = {}
exec(compile(src, "kda_op_bench_tp", "exec"), g)
X = g["X"]
from hip_kda.ops import g2_ops  # noqa: E402

T, H, NT = a.T, a.H, a.T // 64
q, k, v, graw = (X[n].detach() for n in ("q", "k", "v", "g"))
beta, alog, dtb = X["beta"].detach().float(), X["A_log"].detach().float(), X["dt_bias"].detach().float()
scale = 128 ** -0.5
prep = lambda: g2_ops.fwd_prep(q, k, v, graw, alog, dtb, beta, -5.0, 1.4426950216, True, scale, H, NT)
Wn, Kn, Un, QgA, AqkA, Aqk, Akk, qr, kr, Gn = prep()
fwdh = lambda: g2_ops.fwd_h_o(Wn, Kn, Un, Gn, QgA, AqkA, scale, emit_bn=False, o_bf16=True, emit_vn=False)

hip = ctypes.CDLL("libamdhip64.so")
ncu = torch.cuda.get_device_properties(0).multi_processor_count
nw = (ncu + 31) // 32


def masked(lo, hi):
    m = (ctypes.c_uint32 * nw)()
    for c in range(lo, hi):
        m[c // 32] |= 1 << (c % 32)
    s = ctypes.c_void_p()
    rc = hip.hipExtStreamCreateWithCUMask(ctypes.byref(s), ctypes.c_uint32(nw), m)
    assert rc == 0, rc
    return torch.cuda.ExternalStream(s.value)


sA, sB = masked(0, a.rcu), masked(a.rcu, ncu)


def run(pairs):
    """pairs: list of (stream, fn); returns per-fn times and the overall time (us)."""
    torch.cuda.synchronize()
    g0 = torch.cuda.Event(enable_timing=True)
    g0.record()
    evs = []
    for st, fn in pairs:
        st.wait_event(g0)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(st):
            e0.record(); fn(); e1.record()
        evs.append((e0, e1))
    torch.cuda.synchronize()
    per = [e0.elapsed_time(e1) * 1e3 for e0, e1 in evs]
    tot = max(g0.elapsed_time(e1) for _, e1 in evs) * 1e3
    return per, tot


cur = torch.cuda.current_stream()
cases = {"fwdh_all": [(cur, fwdh)], "prep_all": [(cur, prep)], "fwdh_A": [(sA, fwdh)], "prep_B": [(sB, prep)],
         "both": [(sA, fwdh), (sB, prep)]}
res = {c: [] for c in cases}
for r in range(a.reps + 2):
    for c, pr in cases.items():
        per, tot = run(pr)
        if r >= 2:
            res[c].append((per, tot))
med = lambda xs: statistics.median(xs)
for c in cases:
    pers = list(zip(*[x[0] for x in res[c]]))
    print(f"{c:9s} total {med([x[1] for x in res[c]]):8.1f} us   per-stream " + "  ".join(f"{med(p_):8.1f}" for p_ in pers))
