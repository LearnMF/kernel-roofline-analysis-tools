"""B-class precision gate for flash-train trees (R23b rules 1-2): per-tensor rms error vs the cached FP64 oracle
(frozen reference_kda.py, tests/s5/run_fp64_oracle.py inputs), for several builds on identical inputs.
K3 route (in-kernel l2norm + gate; beta sigmoid in torch fp32, identical for every build).
    python3 ft_fp64_gate.py OUT.jsonl name=pythonpath [name=pythonpath ...]   (cases from S5_CASES)"""
import importlib.util, json, math, os, subprocess, sys
CASES = os.environ.get("S5_CASES", "4096:12:101,4096:12:102,4096:12:103,4096:12:104,16384:12:7,32768:4:11").split(",")
ORC = os.environ.get("S5_CACHE", "/public/home/tanbo/s5work/oracle")
TREE = os.environ.get("S5_TREE", "/public/home/tanbo/g2_opt")
CHILD = r'''
import importlib.util, json, math, sys, torch
spec = importlib.util.spec_from_file_location("o", "%s/tests/s5/run_fp64_oracle.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from flash_train.torch import chunk_kda
T, H, seed = %d, %d, %d
orc = torch.load("%s/%%d_%%d_%%d.pt" %% (T, H, seed))
base = m.base_inputs(T, H, seed)
def run():
    x = {n: t.detach().clone().cuda() for n, t in base.items()}
    for n in m.NAMES: x[n].requires_grad_(True)
    beta = torch.sigmoid(x["beta_raw"].float())
    o, _ = chunk_kda(x["q"], x["k"], x["v"], x["g_raw"], beta, scale=1.0 / math.sqrt(128), use_qk_l2norm_in_kernel=True,
                     use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0, A_log=x["A_log"], dt_bias=x["dt_bias"])
    (o.float() * x["d_o"].float()).sum().backward(); torch.cuda.synchronize()
    return o.detach().float().cpu(), {n: x[n].grad.detach().float().cpu() for n in m.NAMES}
o, g = run(); o2, g2 = run()
det = torch.equal(o, o2) and all(torch.equal(g[n], g2[n]) for n in m.NAMES)
r = {"o": m.rms_ratio(o, orc["o"])}
for n in m.NAMES: r[n] = m.rms_ratio(g[n], orc["grads"][n])
print("RES " + json.dumps({"det": det, "rms": r}))
'''
out = open(sys.argv[1], "w")
for case in CASES:
    T, H, seed = map(int, case.split(":"))
    for spec in sys.argv[2:]:
        name, pp = spec.split("=", 1)
        env = dict(os.environ, PYTHONPATH=pp)
        p = subprocess.run([sys.executable, "-c", CHILD % (TREE, T, H, seed, ORC)], env=env, capture_output=True, text=True)
        line = [l for l in p.stdout.splitlines() if l.startswith("RES ")]
        rec = {"name": name, "T": T, "H": H, "seed": seed, **(json.loads(line[0][4:]) if line else {"err": p.stderr[-300:]})}
        out.write(json.dumps(rec) + "\n"); out.flush()
        print(name, T, H, seed, rec.get("det"), {k: f"{v:.3e}" for k, v in rec.get("rms", {}).items()} or rec.get("err"), flush=True)
