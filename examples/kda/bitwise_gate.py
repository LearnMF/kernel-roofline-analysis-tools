"""Bitwise gate for KDA kernel changes: dump o + all 7 input gradients of one fwd+bwd,
or compare two dumps element-for-element (torch.equal, no tolerance).

  dump:    python3 bitwise_gate.py dump --tree TREE --T 8192 --H 12 --out base.pt [--keep-state]
  compare: python3 bitwise_gate.py cmp base.pt cand.pt

Inputs are exactly the K3 bench inputs (kda_op_bench_tp.py, seed T+H). `--keep-state`
runs the forward inside a reentrant checkpoint so the R5 keep-state path (fwd emits h /
v_new, bwd skips fwd_h_bn) is exercised too.
"""
import argparse
import sys

import torch


def dump(a):
    sys.argv = ["x", str(a.T), str(a.H), "g2", "1"]
    src = open(f"{a.tree}/tests/megatron/kda_op_bench_tp.py").read().split("def fwd(arm")[0]
    g = {}
    exec(compile(src, "kda_op_bench_tp", "exec"), g)
    L, D_O, CU, CU_CPU = g["LEAVES"], g["D_O"], g["CU"], g["CU_CPU"]
    if a.cu:             # varlen: packed sequences with the given boundaries (must end at T)
        b = [int(x) for x in a.cu.split(",")]
        assert b[0] == 0 and b[-1] == a.T, "cu boundaries must start at 0 and end at T"
        CU_CPU = torch.tensor(b, dtype=torch.int32)
        CU = CU_CPU.cuda()
    if a.gate_extreme:   # every token's decay saturates at lower_bound: widest exp2 argument ranges
        with torch.no_grad():
            L["g"].mul_(0).add_(8.0)
            L["A_log"].fill_(float(torch.log(torch.tensor(16.0))))
    from hip_kda.fla import chunk_kda
    kw = dict(A_log=L["A_log"], dt_bias=L["dt_bias"], scale=None, output_final_state=False,
              use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=False,
              safe_gate=True, lower_bound=-5.0, cu_seqlens=CU, cu_seqlens_cpu=CU_CPU)
    names = list(L)

    def f(*xs):
        T = dict(zip(names, xs))
        k2 = dict(kw, A_log=T["A_log"], dt_bias=T["dt_bias"])
        return chunk_kda(T["q"], T["k"], T["v"], T["g"], T["beta"], **k2)[0]

    if a.keep_state:   # reentrant checkpoint: forward re-runs inside backward -> R5 keep-state
        from torch.utils.checkpoint import checkpoint
        o = checkpoint(f, *L.values(), use_reentrant=True)
    else:
        o = f(*L.values())
    for t in L.values():
        t.grad = None
    o.backward(D_O)            # (reentrant checkpoint forbids autograd.grad)
    grads = [t.grad for t in L.values()]
    torch.cuda.synchronize()
    res = {"o": o.detach().cpu()}
    res.update({f"d{k}": gr.detach().cpu() for k, gr in zip(L, grads)})
    torch.save(res, a.out)
    print("DUMPED", a.out, {k: tuple(v.shape) for k, v in res.items()})


def cmp(a):
    x, y = torch.load(a.base), torch.load(a.cand)
    ok = True
    for k in x:
        same = torch.equal(x[k], y[k])
        nd = 0 if same else int((x[k] != y[k]).sum())
        ok &= same
        print(f"{k:8s} {'BITWISE' if same else 'DIFF'} {'' if same else f'{nd} elems, max|d|=' + str(float((x[k].float() - y[k].float()).abs().max()))}")
    print("GATE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


p = argparse.ArgumentParser()
sp = p.add_subparsers(dest="cmd", required=True)
d = sp.add_parser("dump")
d.add_argument("--tree", required=True)
d.add_argument("--T", type=int, default=8192)
d.add_argument("--H", type=int, default=12)
d.add_argument("--out", required=True)
d.add_argument("--keep-state", action="store_true")
d.add_argument("--gate-extreme", action="store_true", help="saturate every token's decay at lower_bound")
d.add_argument("--cu", default=None, help="varlen boundaries, e.g. 0,1000,1063,5000,8192")
c = sp.add_parser("cmp")
c.add_argument("base")
c.add_argument("cand")
a = p.parse_args()
dump(a) if a.cmd == "dump" else cmp(a)
