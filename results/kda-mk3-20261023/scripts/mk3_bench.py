"""KDA op bench with the exact Megatron-LM-k3 call (megatron/core/ssm/gated_delta_net/kda.py):
  chunk_kda(q, k, v, g=raw_g, beta, A_log, dt_bias, initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=False, use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0,
            cu_seqlens=cu_seqlens_q, cp_context=None)
q / k arrive l2-normalized (Megatron runs FLA's l2norm before the call), beta is fp32 sigmoid,
g the raw bf16 gate, B = 1 THD with cu_seqlens = [0, T].
  python3 mk3_bench.py T H arms(ft,fla,fla_nr) [reps] [mode: mk3 | k3]
mode k3 = the Megatron-Bridge K3 call we optimized (raw q/k, use_qk_l2norm_in_kernel=True), same tensors.
One JSON line per arm: fwd / bwd / fb wall (median), fb device time (sum of kernels, profiler), peak memory."""
import json, statistics, sys, time
import torch
import torch.nn.functional as F

T, H = int(sys.argv[1]), int(sys.argv[2])
ARMS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["ft", "fla"]
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 7
MODE = sys.argv[5] if len(sys.argv) > 5 else "mk3"
gen = torch.Generator(device="cuda").manual_seed(T + H)
r = lambda *s: torch.randn(*s, device="cuda", generator=gen)
q0, k0 = r(1, T, H, 128), r(1, T, H, 128)
if MODE == "mk3":
    q0, k0 = F.normalize(q0, dim=-1), F.normalize(k0, dim=-1)
X = {"q": q0.bfloat16(), "k": k0.bfloat16(), "v": (r(1, T, H, 128) * .3).bfloat16(),
     "g": (r(1, T, H, 128) * .3).bfloat16(), "beta": r(1, T, H).sigmoid(),
     "A_log": torch.log(1 + 15 * torch.rand(H, device="cuda", generator=gen)), "dt_bias": r(H * 128) * .1}
D_O = r(1, T, H, 128).bfloat16()
CU = torch.tensor([0, T], dtype=torch.int32, device="cuda")
L = {k: v.requires_grad_(True) for k, v in X.items()}
KW = dict(A_log=L["A_log"], dt_bias=L["dt_bias"], initial_state=None, output_final_state=False,
          use_qk_l2norm_in_kernel=(MODE == "k3"), use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0,
          cu_seqlens=CU, cp_context=None)
fns = {}
if "ft" in ARMS:
    import flash_train.torch as ft
    fns["ft"] = lambda: ft.chunk_kda(L["q"], L["k"], L["v"], L["g"], L["beta"], **KW)[0]
if any(a.startswith("fla") for a in ARMS):
    from fla.ops.kda import chunk_kda as fla_kda
    fns["fla"] = lambda: fla_kda(L["q"], L["k"], L["v"], L["g"], L["beta"], **KW)[0]
    fns["fla_nr"] = lambda: fla_kda(L["q"], L["k"], L["v"], L["g"], L["beta"], disable_recompute=True, **KW)[0]


def zero():
    for t in L.values():
        t.grad = None


def fb(f):
    zero()
    f().backward(D_O)


from torch.profiler import ProfilerActivity, profile
for arm in ARMS:
    f = fns[arm]
    for _ in range(3):
        fb(f)
    torch.cuda.synchronize()
    fw, bw = [], []
    for _ in range(REPS):
        zero(); torch.cuda.synchronize()
        t0 = time.perf_counter(); o = f(); torch.cuda.synchronize(); t1 = time.perf_counter()
        o.backward(D_O); torch.cuda.synchronize(); t2 = time.perf_counter()
        fw.append((t1 - t0) * 1e3); bw.append((t2 - t1) * 1e3)
    ks = []
    for _ in range(3):
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            fb(f); torch.cuda.synchronize()
        ks.append(sum(e.device_time_total for e in p.key_averages()
                      if e.device_time_total > 0 and e.device_type.name == "CUDA") / 1e3)
    torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
    fb(f); torch.cuda.synchronize()
    print(json.dumps({"mode": MODE, "T": T, "H": H, "arm": arm,
                      "fwd_ms": round(statistics.median(fw), 4), "bwd_ms": round(statistics.median(bw), 4),
                      "fb_ms": round(statistics.median([a + b for a, b in zip(fw, bw)]), 4),
                      "kernel_fb_ms": round(statistics.mean(ks), 4),
                      "peak_over_inputs_gib": round((torch.cuda.max_memory_allocated() - base) / 2**30, 3)}), flush=True)
