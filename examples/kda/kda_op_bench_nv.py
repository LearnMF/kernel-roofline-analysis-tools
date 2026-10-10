"""NVIDIA counterpart of hip_kda's tests/megatron/kda_op_bench_tp.py (the BW operator bench):
the SAME Kimi K3 KDA call, inputs, arms, metrics and checkpoint mode, with the DCU-only pieces
removed (no fla_prune_ns4 config pruning -- FLA's own NV autotune configs are the natural
competitor setup; no hip_kda / flash-train arms).

    python3 kda_op_bench_nv.py T H [arms=fla,fla_nr] [reps=7]      (BENCH_CKPT=1: + Megatron form)

Inputs follow KimiK3Attention._forward_kda: q/k/v bf16 [1,T,H,128] (leaves), raw gate g bf16 with
A_log [H] / dt_bias [H*128] fp32 (lower_bound -5, gate activated in the kernel), beta fp32
post-sigmoid, qk l2norm in the kernel, cu_seqlens [0, T].  Metrics per arm: fwd / bwd / fb wall
(events, median of reps), kernel_fb_ms (summed device time of one F+B, profiler, mean of 3),
peak / saved memory, and with BENCH_CKPT=1 the reentrant-checkpoint step (wall + kernel time).
`--dump FILE` / `--load FILE`: save or reuse the exact inputs (cross-platform numerics check).
"""
import json
import os
import statistics
import sys

import torch

args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = {a.split("=")[0]: (a.split("=", 1)[1] if "=" in a else True) for a in sys.argv[1:] if a.startswith("--")}
T, H = int(args[0]), int(args[1])
ARMS = (args[2] if len(args) > 2 else "fla,fla_nr").split(",")
REPS = int(args[3]) if len(args) > 3 else 7

if os.environ.get("FLA_PRUNE_NS4") == "1":   # BW/DCU only: the same autotune pruning as the BW bench
    import fla_prune_ns4  # noqa: E402

    fla_prune_ns4.apply(verbose=False)
from fla.ops.kda import chunk_kda as fla_chunk_kda  # noqa: E402

gen = torch.Generator(device="cuda").manual_seed(T + H)
r = lambda *s: torch.randn(*s, device="cuda", generator=gen)
X = {"q": (r(1, T, H, 128) * .3).bfloat16(), "k": (r(1, T, H, 128) * .3).bfloat16(),
     "v": (r(1, T, H, 128) * .3).bfloat16(), "g": (r(1, T, H, 128) * .3).bfloat16(),
     "beta": r(1, T, H).sigmoid(), "A_log": torch.log(1 + 15 * torch.rand(H, device="cuda", generator=gen)),
     "dt_bias": r(H * 128) * .1}
D_O = r(1, T, H, 128).bfloat16()
if "--load" in opts:
    blob = torch.load(opts["--load"], map_location="cuda")
    X = {k: blob[k].cuda() for k in X}
    D_O = blob["d_o"].cuda()
CU = torch.tensor([0, T], dtype=torch.int32, device="cuda")
CU_CPU = CU.cpu()
LEAVES = {k: v.requires_grad_(True) for k, v in X.items()}


def fwd(arm, L=None):
    L = LEAVES if L is None else L
    kw = dict(A_log=L["A_log"], dt_bias=L["dt_bias"], scale=None, output_final_state=False,
              use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=False,
              safe_gate=True, lower_bound=-5.0, cu_seqlens=CU, cu_seqlens_cpu=CU_CPU)
    if arm == "fla":
        return fla_chunk_kda(L["q"], L["k"], L["v"], L["g"], L["beta"], **kw)[0]
    if arm == "fla_nr":
        return fla_chunk_kda(L["q"], L["k"], L["v"], L["g"], L["beta"], disable_recompute=True, **kw)[0]
    raise ValueError(arm)


def bwd(o):
    torch.autograd.grad(o, list(LEAVES.values()), D_O)


if "--dump" in opts:      # o + all 7 grads of the default arm, for a cross-platform comparison
    o = fwd(ARMS[0])
    grads = torch.autograd.grad(o, list(LEAVES.values()), D_O)
    out = {k: v.detach().cpu() for k, v in X.items()}
    out["d_o"] = D_O.cpu()
    out["o"] = o.detach().cpu()
    out.update({f"d{k}": g.detach().cpu() for k, g in zip(LEAVES, grads)})
    torch.save(out, opts["--dump"])
    print(json.dumps({"tag": "dump", "file": opts["--dump"]}))
    sys.exit(0)


def ev():
    return torch.cuda.Event(enable_timing=True)


def time_arm(arm):
    s, m, e = ev(), ev(), ev()
    s.record(); o = fwd(arm); m.record(); bwd(o); e.record()
    torch.cuda.synchronize()
    f, fb = s.elapsed_time(m), s.elapsed_time(e)
    o = fwd(arm); torch.cuda.synchronize()
    s2, e2 = ev(), ev()
    s2.record(); bwd(o); e2.record(); torch.cuda.synchronize()
    del o
    return f, s2.elapsed_time(e2), fb


def mem_arm(arm):
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    o = fwd(arm); torch.cuda.synchronize()
    saved = torch.cuda.memory_allocated() - base
    bwd(o); del o; torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(), saved


def _kernel_sum(p):
    return sum(e.device_time_total for e in p.key_averages()
               if e.device_type == torch.autograd.DeviceType.CUDA) / 3000.0


def kernel_ms(arm):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(3):
            o = fwd(arm); bwd(o); del o
        torch.cuda.synchronize()
    return _kernel_sum(p)


CKPT = os.environ.get("BENCH_CKPT", "0") == "1"
KEYS = list(LEAVES)


def ckpt_step(arm):
    """Megatron full recompute: reentrant checkpoint = no-grad forward + recompute forward
    inside the backward + backward (Megatron's CheckpointFunction form)."""
    import torch.utils.checkpoint as tuc
    o = tuc.checkpoint(lambda *t: fwd(arm, dict(zip(KEYS, t))), *LEAVES.values(), use_reentrant=True)
    o.backward(D_O)
    for t in LEAVES.values():
        t.grad = None


def ckpt_kernel_ms(arm):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(3):
            ckpt_step(arm)
        torch.cuda.synchronize()
    return _kernel_sum(p)


res, live = {}, []
for arm in ARMS:
    try:
        for _ in range(2):
            o = fwd(arm); bwd(o); del o
        torch.cuda.synchronize()
        peak, saved = mem_arm(arm)
        res[arm] = {"peak_GiB": round(peak / 2**30, 2), "saved_B_per_tok_head": round(saved / (T * H)), "t": []}
        live.append(arm)
    except torch.cuda.OutOfMemoryError:
        res[arm] = {"oom": True}
        torch.cuda.empty_cache()
for i in range(REPS):
    for arm in (live if i % 2 == 0 else live[::-1]):
        res[arm]["t"].append(time_arm(arm))
for arm in live:
    t = res[arm].pop("t")
    res[arm].update(fwd_ms=round(statistics.median(x[0] for x in t), 3),
                    bwd_ms=round(statistics.median(x[1] for x in t), 3),
                    fb_ms=round(statistics.median(x[2] for x in t), 3))
    res[arm]["kernel_fb_ms"] = round(kernel_ms(arm), 3)
if CKPT:
    ck = {}
    for arm in live:
        for _ in range(2):
            ckpt_step(arm)
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        ckpt_step(arm); torch.cuda.synchronize()
        res[arm]["ckpt_peak_GiB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        ck[arm] = []
    for i in range(REPS):
        for arm in (list(ck) if i % 2 == 0 else list(ck)[::-1]):
            s0, e0 = ev(), ev()
            s0.record(); ckpt_step(arm); e0.record(); torch.cuda.synchronize()
            ck[arm].append(s0.elapsed_time(e0))
    for arm, t in ck.items():
        res[arm]["ckpt_step_ms"] = round(statistics.median(t), 3)
        res[arm]["ckpt_kernel_ms"] = round(ckpt_kernel_ms(arm), 3)
import fla  # noqa: E402
import triton  # noqa: E402

env = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "triton": triton.__version__,
       "fla": getattr(fla, "__version__", "?"), "cuda": torch.version.cuda}
print(json.dumps({"tag": "kda_op_bench", "T": T, "H": H, "tp": 96 // H, "env": env, "res": res}), flush=True)
