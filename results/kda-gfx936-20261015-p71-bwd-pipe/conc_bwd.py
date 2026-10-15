"""C-5: contention-only bound (as C-4 / conc_mask.py) for the backward overlap pairs of plan A:
    rec || wu   the recompute recurrence (fwd_h_bn) next to its chunk-parallel producer (wu)
    rec || dav  ... next to its chunk-parallel consumer (dav)
    dhu || s4   the dh recurrence (dhu_bn) next to its chunk-parallel consumer (wy_s4_split = s4a + s4b)
The real backward runs once with g2_ops wrappers that keep each call's arguments; every case then replays the
recorded calls: alone on all CUs, alone on its CU partition (recurrence: RCU CUs, partner: the rest), and both
partitions concurrently (independent operands, no flags).
    PYTHONPATH=<env>:<tree>:<takeover> python3 conc_bwd.py --tree TREE --T 16384 --H 12 [--rcu 48]"""
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
p.add_argument("--reps", type=int, default=10)
a = p.parse_args()
sys.argv = ["x", str(a.T), str(a.H), "g2", "1"]
src = open(f"{a.tree}/tests/megatron/kda_op_bench_tp.py").read().split("def fwd(arm")[0]
g = {}
exec(compile(src, "kda_op_bench_tp", "exec"), g)
L, D_O, CU, CU_CPU = g["LEAVES"], g["D_O"], g["CU"], g["CU_CPU"]
from hip_kda.fla import chunk_kda  # noqa: E402
from hip_kda.ops import g2_ops  # noqa: E402

REC = {}
for name in ("wu", "fwd_h_bn", "dav", "dhu_bn", "wy_s4_split"):
    fn = getattr(g2_ops, name)

    def wrap(*args, _fn=fn, _n=name, **kw):
        REC[_n] = (_fn, args, kw)
        return _fn(*args, **kw)
    setattr(g2_ops, name, wrap)

kw = dict(scale=None, output_final_state=False, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
          use_beta_sigmoid_in_kernel=False, safe_gate=True, lower_bound=-5.0, cu_seqlens=CU, cu_seqlens_cpu=CU_CPU)
out = chunk_kda(L["q"], L["k"], L["v"], L["g"], L["beta"], A_log=L["A_log"], dt_bias=L["dt_bias"], **kw)
o = out[0] if isinstance(out, (tuple, list)) else out
o.backward(D_O)
torch.cuda.synchronize()
assert all(n in REC for n in ("wu", "fwd_h_bn", "dav", "dhu_bn", "wy_s4_split")), sorted(REC)
call = {n: (lambda n=n: REC[n][0](*REC[n][1], **REC[n][2])) for n in REC}

hip = ctypes.CDLL("libamdhip64.so")
ncu = torch.cuda.get_device_properties(0).multi_processor_count
nw = (ncu + 31) // 32


def masked(lo, hi):
    m = (ctypes.c_uint32 * nw)()
    for c in range(lo, hi):
        m[c // 32] |= 1 << (c % 32)
    s = ctypes.c_void_p()
    assert hip.hipExtStreamCreateWithCUMask(ctypes.byref(s), ctypes.c_uint32(nw), m) == 0
    return torch.cuda.ExternalStream(s.value)


sA, sB, cur = masked(0, a.rcu), masked(a.rcu, ncu), torch.cuda.current_stream()


def run(pairs):
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
    return [e0.elapsed_time(e1) * 1e3 for e0, e1 in evs]


for rec, partner in (("fwd_h_bn", "wu"), ("fwd_h_bn", "dav"), ("dhu_bn", "wy_s4_split")):
    cases = {"rec_all": [(cur, call[rec])], "rec_A": [(sA, call[rec])],
             "par_all": [(cur, call[partner])], "par_B": [(sB, call[partner])],
             "both": [(sA, call[rec]), (sB, call[partner])]}
    res = {c: [] for c in cases}
    for r in range(a.reps + 2):
        for c, pr in cases.items():
            t = run(pr)
            if r >= 2:
                res[c].append(t)
    m = {c: [statistics.median(x) for x in zip(*v)] for c, v in res.items()}
    R, Rc, P, PB = m["rec_A"][0], m["both"][0], m["par_all"][0], m["par_B"][0]
    cap = min(P, Rc * P / PB)    # full-GPU-equivalent partner work done during the recurrence
    print(f"{rec:9s} || {partner:11s}: rec all/A/conc {m['rec_all'][0]:7.1f}/{R:7.1f}/{Rc:7.1f} us "
          f"(+{100 * (Rc / R - 1):.0f}%)  partner all/B/conc {P:7.1f}/{PB:7.1f}/{m['both'][1]:7.1f} us "
          f"(B slowdown {PB / P:.2f}x)  bound = cap {cap:6.1f} - rec slowdown {Rc - R:6.1f} = {cap - (Rc - R):6.1f} us "
          f"(before hand-offs)")
