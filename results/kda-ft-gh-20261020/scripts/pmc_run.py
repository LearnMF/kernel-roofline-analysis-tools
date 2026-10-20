import os, sys, torch
sys.argv = ["x", sys.argv[1], sys.argv[2], os.environ["ARMS"], "1"]
src = open("/public/home/tanbo/xplat/kda_op_bench_ftgh.py").read().split("res, live = {}, []")[0]
g = {}; exec(compile(src, "b", "exec"), g)
arm = os.environ["ARMS"]
for _ in range(4): o = g["fwd"](arm); g["bwd"](o); del o
torch.cuda.synchronize(); print("PMC_RUN_DONE", arm)
