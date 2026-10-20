"""Operator-level KDA benchmark on the Kimi K3 call (one KDA layer, per card under TP).

    python3 kda_op_bench_tp.py T H [arms] [reps]

K3: 96 KDA heads, so TP = 1/2/4/8 gives H = 96/48/24/12 heads per card. Inputs follow
KimiK3Attention._forward_kda: q/k/v bf16 [1,T,H,128] (leaves here), raw gate g bf16 with
A_log [H] / dt_bias [H*128] fp32 (lower_bound -5, gate activated in the kernel), beta fp32
post-sigmoid, qk l2norm in the kernel, cu_seqlens [0, T].

Arms (same inputs, same process, interleaved per repetition):
  fla      FLA chunk_kda default (recompute in the backward), DCU-pruned configs
  fla_nr   FLA chunk_kda disable_recompute=True (saves the forward intermediates)
  g2       hip_kda G2 through hip_kda.fla.chunk_kda -- the entry Megatron/Bridge calls
  ft       flash-train flash_train.torch.kimi_delta_attention (same kernels as g2)

Per arm: forward (training forward, autograd graph built), backward alone (forward untimed),
forward+backward -- median of `reps`; kernel_fb_ms: the summed device time of all kernels in
one forward+backward (profiler, mean of 3) -- what a GPU-bound training step sees, without the
host-side launch overhead that wall-clock timing of an isolated call includes; memory: peak allocated during one F+B with the inputs
and d_o already resident (absolute GiB, the "peak HBM"), and the bytes the forward leaves
allocated for the backward (incl. o) per token*head.
"""
import json
import os
import statistics
import sys

import torch

T, H = int(sys.argv[1]), int(sys.argv[2])
ARMS = (sys.argv[3] if len(sys.argv) > 3 else "fla,fla_nr,g2,ft").split(",")
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 7
os.environ.setdefault("KDA_G2", "1")

import fla_prune_ns4  # noqa: E402

fla_prune_ns4.apply(verbose=False)
from fla.ops.kda import chunk_kda as fla_chunk_kda  # noqa: E402

from hip_kda.fla import chunk_kda as g2_chunk_kda  # noqa: E402

try:
    import flash_train.torch as ft  # noqa: E402
except Exception:  # noqa: BLE001
    ft = None
    ARMS = [a for a in ARMS if a not in ("ft", "ftc")]

gen = torch.Generator(device="cuda").manual_seed(T + H)
r = lambda *s: torch.randn(*s, device="cuda", generator=gen)
X = {"q": (r(1, T, H, 128) * .3).bfloat16(), "k": (r(1, T, H, 128) * .3).bfloat16(),
     "v": (r(1, T, H, 128) * .3).bfloat16(), "g": (r(1, T, H, 128) * .3).bfloat16(),
     "beta": r(1, T, H).sigmoid(), "A_log": torch.log(1 + 15 * torch.rand(H, device="cuda", generator=gen)),
     "dt_bias": r(H * 128) * .1}
D_O = r(1, T, H, 128).bfloat16()
CU = None if os.environ.get("BENCH_DENSE") == "1" else torch.tensor([0, T], dtype=torch.int32, device="cuda")
CU_CPU = CU.cpu() if CU is not None else None
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
    if arm == "g2":
        return g2_chunk_kda(L["q"], L["k"], L["v"], L["g"], L["beta"], **kw)[0]
    if arm == "ftc":  # flash_train.torch.chunk_kda: the FLA-signature entry Megatron calls
        return ft.chunk_kda(L["q"], L["k"], L["v"], L["g"], L["beta"], **kw)[0]
    o = ft.kimi_delta_attention(L["q"][0], L["k"][0], L["v"][0], L["g"][0], L["beta"][0], A_log=L["A_log"],
                                dt_bias=L["dt_bias"], cu_seqlens=CU)
    return o.unsqueeze(0)


def bwd(o):
    torch.autograd.grad(o, list(LEAVES.values()), D_O)


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


def kernel_ms(arm):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(3):
            o = fwd(arm); bwd(o); del o
        torch.cuda.synchronize()
    return sum(e.device_time_total for e in p.key_averages()
               if e.device_type == torch.autograd.DeviceType.CUDA) / 3000.0


CKPT = os.environ.get("BENCH_CKPT", "0") == "1"
KEYS = list(LEAVES)


def ckpt_step(arm):
    """One Megatron full-recompute step of this layer: reentrant checkpoint = no-grad forward +
    recompute forward inside the backward + backward (Megatron's CheckpointFunction form)."""
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
    return sum(e.device_time_total for e in p.key_averages()
               if e.device_type == torch.autograd.DeviceType.CUDA) / 3000.0


res, live = {}, []
for arm in ARMS:
    try:
        for _ in range(2):
            o = fwd(arm); bwd(o); del o
        torch.cuda.synchronize()
        peak, saved = mem_arm(arm)
        res[arm] = {"peak_GiB": round(peak / 2**30, 2), "saved_B_per_tok_head": round(saved / (T * H)),
                    "t": []}
        live.append(arm)
    except torch.cuda.OutOfMemoryError:
        res[arm] = {"oom": True}
        torch.cuda.empty_cache()
for i in range(REPS):
    for arm in (live if i % 2 == 0 else live[::-1]):
        try:
            res[arm]["t"].append(time_arm(arm))
        except torch.cuda.OutOfMemoryError:
            res[arm]["oom_timing"] = True
            torch.cuda.empty_cache()
for arm in live:
    t = res[arm].pop("t")
    if t:
        res[arm].update(fwd_ms=round(statistics.median(x[0] for x in t), 3),
                        bwd_ms=round(statistics.median(x[1] for x in t), 3),
                        fb_ms=round(statistics.median(x[2] for x in t), 3))
for arm in live:
    try:
        res[arm]["kernel_fb_ms"] = round(kernel_ms(arm), 3)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
if CKPT:
    for arm in live:
        try:
            for _ in range(2):
                ckpt_step(arm)
            torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            ckpt_step(arm); torch.cuda.synchronize()
            res[arm]["ckpt_peak_GiB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        except torch.cuda.OutOfMemoryError:
            res[arm]["ckpt_oom"] = True
            torch.cuda.empty_cache()
    ck = {a: [] for a in live if not res[a].get("ckpt_oom")}
    for i in range(REPS):
        for arm in (list(ck) if i % 2 == 0 else list(ck)[::-1]):
            s0, e0 = ev(), ev()
            s0.record(); ckpt_step(arm); e0.record(); torch.cuda.synchronize()
            ck[arm].append(s0.elapsed_time(e0))
    for arm, t in ck.items():
        res[arm]["ckpt_step_ms"] = round(statistics.median(t), 3)
        res[arm]["ckpt_kernel_ms"] = round(ckpt_kernel_ms(arm), 3)
inputs_GiB = round(sum(v.numel() * v.element_size() for v in X.values()) / 2**30 + D_O.numel() * 2 / 2**30, 2)
print(json.dumps({"tag": "kda_op_bench", "T": T, "H": H, "tp": 96 // H, "inputs_GiB": inputs_GiB, "res": res}),
      flush=True)
