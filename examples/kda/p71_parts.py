"""P-71 cost breakdown: recompute with / without progress publishing, dav head-major vs chunk-major, on all CUs and
on the CU partitions, and the pipelined pair without the hand-off (both masked streams, part 1 = all chunks).
    PYTHONPATH=<env>:<P-71 tree>:<takeover> python3 p71_parts.py --tree TREE --T 16384 --H 12"""
import argparse
import ctypes
import statistics
import sys

import torch

p = argparse.ArgumentParser()
p.add_argument("--tree", required=True)
p.add_argument("--T", type=int, default=16384)
p.add_argument("--H", type=int, default=12)
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
for name in ("fwd_h_bn", "dav"):
    fn = getattr(g2_ops, name)

    def wrap(*args, _fn=fn, _n=name, **kw):
        REC[_n] = (_fn, args, kw)
        return _fn(*args, **kw)
    setattr(g2_ops, name, wrap)
kw = dict(scale=None, output_final_state=False, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
          use_beta_sigmoid_in_kernel=False, safe_gate=True, lower_bound=-5.0, cu_seqlens=CU, cu_seqlens_cpu=CU_CPU)
out = chunk_kda(L["q"], L["k"], L["v"], L["g"], L["beta"], A_log=L["A_log"], dt_bias=L["dt_bias"], **kw)
(out[0] if isinstance(out, (tuple, list)) else out).backward(D_O)
torch.cuda.synchronize()
ext = g2_ops._ext()
H, NT = a.H, a.T // 64
fr, ra, rk = REC["fwd_h_bn"]
fd, da, dk = REC["dav"]
do_, Vn, Aqk, scale = da[0].contiguous(), da[1], da[2], da[3]
dA = torch.empty(1, 64 * NT, H, 64, dtype=torch.float32, device="cuda")
dVn = torch.empty(1, H, NT, 4, 8, 4, 16, 4, dtype=torch.bfloat16, device="cuda")
dOT = torch.empty(1, H, NT, 8, 4, 4, 16, 4, dtype=torch.bfloat16, device="cuda")
none_i = torch.empty(0, dtype=torch.int32, device="cuda")
prog_full = torch.full((H * 16,), NT + 1, dtype=torch.int32, device="cuda")
prog = torch.zeros(H * 16, dtype=torch.int32, device="cuda")


def rec(pub=False):
    if pub:
        ext.rec_prog_set(prog, 16)
    r = fr(*ra, **rk)
    if pub:
        ext.rec_prog_set(none_i, 0)
    return r


def dav(cm, wait=False):
    ext.dav_part(do_, Vn, Aqk, scale, H, NT, dA, dVn, dOT, 0, NT, cm, prog_full if wait else none_i)


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


sA, sB, cur = masked(0, 48), masked(48, ncu), torch.cuda.current_stream()


def run(pairs):
    prog.zero_()
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


cases = {
    "rec all": [(cur, rec)], "rec all +pub": [(cur, lambda: rec(True))],
    "rec A": [(sA, rec)], "rec A +pub": [(sA, lambda: rec(True))],
    "dav all hm": [(cur, lambda: dav(0))], "dav all cm": [(cur, lambda: dav(1))],
    "dav all cm+wait(ready)": [(cur, lambda: dav(1, True))],
    "dav B hm": [(sB, lambda: dav(0))], "dav B cm": [(sB, lambda: dav(1))],
    "dav all blk": [(cur, lambda: dav(2))], "dav B blk": [(sB, lambda: dav(2))],
    "conc rec A | dav B blk": [(sA, rec), (sB, lambda: dav(2))],
    "conc rec A+pub | dav B blk+wait": [(sA, lambda: rec(True)), (sB, lambda: ext.dav_part(
        do_, Vn, Aqk, scale, H, NT, dA, dVn, dOT, 0, NT, 2, prog))],
    "conc rec A | dav B hm": [(sA, rec), (sB, lambda: dav(0))],
    "conc rec A | dav B cm": [(sA, rec), (sB, lambda: dav(1))],
    "conc rec A+pub | dav B cm+wait": [(sA, lambda: rec(True)), (sB, lambda: ext.dav_part(
        do_, Vn, Aqk, scale, H, NT, dA, dVn, dOT, 0, NT, 1, prog))],
}
res = {c: [] for c in cases}
for r in range(a.reps + 2):
    for c, pr in cases.items():
        t = run(pr)
        if r >= 2:
            res[c].append(t)
for c, v in res.items():
    print(f"{c:32s} " + "  ".join(f"{statistics.median(x):8.1f}" for x in zip(*v)))
