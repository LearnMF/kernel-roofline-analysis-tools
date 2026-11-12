import flash_attn.utils.sparse_utils  # noqa: F401  (order fix for fla import segfault)
import sys

import torch

sys.argv = ["x", "8192", "12", "g2", "1"]
src = open("/opt/kda_env/g2_r5/tests/megatron/kda_op_bench_tp.py").read().split("res, live = {}, []")[0]
g = {}
print("exec...", flush=True)
exec(compile(src, "kda_op_bench_tp", "exec"), g)
print("exec ok", flush=True)
for i in range(2):
    print(f"iter {i} fwd", flush=True)
    o = g["fwd"]("g2")
    torch.cuda.synchronize()
    print(f"iter {i} bwd", flush=True)
    g["bwd"](o)
    del o
    torch.cuda.synchronize()
    print(f"iter {i} done", flush=True)
print("ALL OK", flush=True)
