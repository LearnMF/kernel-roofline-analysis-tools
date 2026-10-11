"""KDA forward+backward driver for timeline capture (L1).

Runs `iters` measured fwd+bwd iterations of one arm of the Kimi K3 KDA op bench,
each preceded by a marker kernel (torch.cuda._sleep -> a kernel whose name contains
"sleep") so `kra timeline --marker sleep` can cut the trace into per-iteration windows.

Usage (inside the KDA container, under hipprof):
    hipprof --hip-trace --output-type 0 -o OUT \
        python3 kda_iter.py --tree /opt/kda_env/g2_r5 --arm g2 --T 8192 --H 12 --iters 5

`--sync` inserts a device synchronize between fwd and bwd (default: no sync, like training).
`--time` (run WITHOUT hipprof) prints the unprofiled per-iteration time measured with
events around the same fwd+bwd window -- the reference for `kra timeline --reference-ms`.
"""
import argparse
import sys

import torch

p = argparse.ArgumentParser()
p.add_argument("--tree", default="/opt/kda_env/g2_r5")
p.add_argument("--arm", default="g2", help="g2 | fla | fla_nr")
p.add_argument("--T", type=int, default=8192)
p.add_argument("--H", type=int, default=12)
p.add_argument("--iters", type=int, default=5)
p.add_argument("--warmup", type=int, default=3)
p.add_argument("--sync", action="store_true")
p.add_argument("--time", action="store_true", help="print unprofiled per-iteration ms")
p.add_argument("--optrace", default=None,
               help="write launcher interface capture (kra.opspec.capture) of the last iteration here")
p.add_argument("--keep-state", action="store_true",
               help="g2 arm under a reentrant checkpoint (Megatron's form): the forward re-runs inside "
                    "the backward and keeps its state, so wu skips the P1 emits and fwd_h_bn is not called")
a = p.parse_args()

# kda_op_bench_tp.py defines the inputs and fwd()/bwd() closures for each arm before
# its own timing loop starts at "res, live = {}, []".
# Placement jitter (rule R20): a per-process dummy allocation shifts every later tensor's
# address, so the A and B sides of an A/B sample the SAME distribution of HBM placements
# instead of one fixed placement per tree (a removed/added allocation otherwise moves all
# other tensors and biases bandwidth-bound kernels by a few %).  KRA_JITTER_SEED=<round>.
import os  # noqa: E402
_jit = os.environ.get("KRA_JITTER_SEED")
if _jit:
    _jit_mb = (int(_jit) * 7919 + 13) % 29 * 8 + 1
    _jitter_buf = torch.empty(_jit_mb * 2**20, dtype=torch.uint8, device="cuda")
sys.argv = ["x", str(a.T), str(a.H), a.arm, "1"]
src = open(f"{a.tree}/tests/megatron/kda_op_bench_tp.py").read().split("res, live = {}, []")[0]
g = {}
exec(compile(src, "kda_op_bench_tp", "exec"), g)
if a.keep_state:
    assert a.arm == "g2", "--keep-state is the G2 route's checkpoint mode"
    from torch.utils.checkpoint import checkpoint
    from hip_kda.fla import chunk_kda
    _L = g["LEAVES"]
    _kw = dict(scale=None, output_final_state=False, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
               use_beta_sigmoid_in_kernel=False, safe_gate=True, lower_bound=-5.0,
               cu_seqlens=g["CU"], cu_seqlens_cpu=g["CU_CPU"])
    _names = list(_L)

    def _f(*xs):
        t = dict(zip(_names, xs))
        return chunk_kda(t["q"], t["k"], t["v"], t["g"], t["beta"], A_log=t["A_log"], dt_bias=t["dt_bias"],
                         **_kw)[0]

    def _fwd(arm):
        return checkpoint(_f, *_L.values(), use_reentrant=True)

    def _bwd(o):
        o.backward(g["D_O"])

    g["fwd"], g["bwd"] = _fwd, _bwd

for _ in range(a.warmup):
    o = g["fwd"](a.arm)
    g["bwd"](o)
    del o
torch.cuda.synchronize()
cap = None
if a.optrace:
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    import hip_kda.ops.g2_ops as g2_ops
    from kra.opspec.capture import LauncherCapture
    cap = LauncherCapture(g2_ops)
times = []
for it in range(a.iters):
    torch.cuda._sleep(200000)          # marker kernel; analysis windows start after it
    torch.cuda.synchronize()
    last = it == a.iters - 1
    if cap is not None and last:
        cap.__enter__()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    o = g["fwd"](a.arm)
    if a.sync:
        torch.cuda.synchronize()
    g["bwd"](o)
    e1.record()
    del o
    torch.cuda.synchronize()
    if cap is not None and last:
        cap.__exit__(None, None, None)
        cap.dump(a.optrace, meta={"T": a.T, "H": a.H, "arm": a.arm, "tree": a.tree})
    times.append(e0.elapsed_time(e1))
if a.time:
    s = sorted(times)
    print(f"KDA_ITER_MS median={s[len(s) // 2]:.4f} min={s[0]:.4f} max={s[-1]:.4f} n={len(s)}")
print("KDA_ITER_DONE", a.arm, a.T, a.H, a.iters)
