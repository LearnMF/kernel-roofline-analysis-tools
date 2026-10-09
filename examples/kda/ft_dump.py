"""Dump o + 7 grads of one flash_train.torch.kimi_delta_attention fwd+bwd on the K3 bench
inputs (same seed / shapes as kda_op_bench_tp.py), for bitwise comparison with
bitwise_gate.py dumps (`python3 bitwise_gate.py cmp a.pt b.pt`).

  PYTHONPATH=<ft_site>:... python3 ft_dump.py --tree /opt/kda_env/g2_r5 --T 8192 --H 12 --out ft.pt [--cu 0,1000,8192]
"""
import argparse
import sys

import torch

p = argparse.ArgumentParser()
p.add_argument("--tree", required=True)
p.add_argument("--T", type=int, default=8192)
p.add_argument("--H", type=int, default=12)
p.add_argument("--out", required=True)
p.add_argument("--cu", default=None)
a = p.parse_args()
sys.argv = ["x", str(a.T), str(a.H), "ft", "1"]
src = open(f"{a.tree}/tests/megatron/kda_op_bench_tp.py").read().split("def fwd(arm")[0]
g = {}
exec(compile(src, "kda_op_bench_tp", "exec"), g)
L, D_O, CU = g["LEAVES"], g["D_O"], g["CU"]
if a.cu:
    CU = torch.tensor([int(x) for x in a.cu.split(",")], dtype=torch.int32, device="cuda")
import flash_train.torch as ft  # noqa: E402

o = ft.kimi_delta_attention(L["q"][0], L["k"][0], L["v"][0], L["g"][0], L["beta"][0], A_log=L["A_log"],
                            dt_bias=L["dt_bias"], cu_seqlens=CU).unsqueeze(0)
for t in L.values():
    t.grad = None
o.backward(D_O)
torch.cuda.synchronize()
res = {"o": o.detach().cpu()}
res.update({f"d{k}": t.grad.detach().cpu() for k, t in L.items()})
torch.save(res, a.out)
print("DUMPED", a.out, ft.__file__)
