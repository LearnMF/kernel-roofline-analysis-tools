"""P-77 unit check: each folded consumer (wu, wy_s4_split, bwd_tail) with raw q/k + rstd vs the same call with
l2n_apply's qn/kn -- bitwise per output.  Records the production calls of one F+B (fold off) and replays them.
    PYTHONPATH=<env>:<tree>:<takeover> KDA_G2_L2FOLD=0 python3 p77_unit.py --tree TREE [--T 8192 --H 12]"""
import argparse
import sys

import torch

p = argparse.ArgumentParser()
p.add_argument("--tree", required=True)
p.add_argument("--T", type=int, default=8192)
p.add_argument("--H", type=int, default=12)
a = p.parse_args()
sys.argv = ["x", str(a.T), str(a.H), "g2", "1"]
src = open(f"{a.tree}/tests/megatron/kda_op_bench_tp.py").read().split("def fwd(arm")[0]
g = {}
exec(compile(src, "kda_op_bench_tp", "exec"), g)
L, D_O, CU, CU_CPU = g["LEAVES"], g["D_O"], g["CU"], g["CU_CPU"]
from hip_kda.fla import chunk_kda  # noqa: E402
from hip_kda.ops import g2_ops  # noqa: E402

REC = {}
for name in ("l2n_apply", "wu", "wy_s4_split", "bwd_tail"):
    f0 = getattr(g2_ops, name)

    def mk(f0, name):
        def w(*args, **kw):
            REC[name] = (args, kw, f0)
            return f0(*args, **kw)
        return w
    setattr(g2_ops, name, mk(f0, name))
kw = dict(scale=None, output_final_state=False, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
          use_beta_sigmoid_in_kernel=False, safe_gate=True, lower_bound=-5.0, cu_seqlens=CU, cu_seqlens_cpu=CU_CPU)
out = chunk_kda(L["q"], L["k"], L["v"], L["g"], L["beta"], A_log=L["A_log"], dt_bias=L["dt_bias"], **kw)
(out[0] if isinstance(out, (tuple, list)) else out).backward(D_O)
torch.cuda.synchronize()
(q, k, qr, kr), _, l2f = REC["l2n_apply"]
qn, kn = l2f(q, k, qr, kr)


def flat(o):
    o = o if isinstance(o, (tuple, list)) else [o]
    return [t for t in o if isinstance(t, torch.Tensor) and t.numel()]


def bits(t):
    return t.view(torch.int16) if t.dtype in (torch.bfloat16, torch.float16) else t


def cmp(name, o1, o2):
    r = []
    for i, (x, y) in enumerate(zip(flat(o1), flat(o2))):
        same = torch.equal(bits(x), bits(y))
        r.append(f"o{i}:{'=' if same else 'DIFF(%d)' % int((bits(x) != bits(y)).sum())}")
    print(f"{name}: {' '.join(r)}")


args, kws, f = REC["wu"]
ref = f(*args, **kws)
args2 = list(args)
args2[0], args2[2] = k, q                      # wu(k, v, q, ...)
cmp("wu", ref, f(*args2, **dict(kws, q_rstd=qr, k_rstd=kr)))
args, kws, f = REC["wy_s4_split"]
ref = f(*args, **kws)
args2 = list(args)
args2[1], args2[11] = k.reshape(-1), q.reshape(-1)   # (q_kgb, k_row, ..., q_row(11), ...)
cmp("wy_s4_split", ref, f(*args2, **dict(kws, q_rstd=qr, k_rstd=kr)))
args, kws, f = REC["bwd_tail"]
ref = f(*args, **kws)
args2 = list(args)
args2[4], args2[5] = q, k                      # bwd_tail(dq, dk, dg, db, qn, kn, ...)
cmp("bwd_tail", ref, f(*args2, **dict(kws, raw_y=True)))
print("qn==l2n(q):", torch.equal(bits(qn), bits(REC["wu"][0][2])))
