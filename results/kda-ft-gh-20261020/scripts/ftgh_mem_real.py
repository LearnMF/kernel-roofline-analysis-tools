"""Peak memory of one KDA F+B where q/k/v/g are produced by an upstream op (x * 1.0, which saves
nothing) instead of being leaves the caller keeps -- the Megatron layer situation: whatever KDA does
not save is freed after the forward. Args: T H"""
import os, sys, torch
T, H = int(sys.argv[1]), int(sys.argv[2])
import fla_prune_ns4; fla_prune_ns4.apply(verbose=False)
from hip_kda.fla import chunk_kda as g2
import flash_train.torch as ft
gen = torch.Generator(device="cuda").manual_seed(T + H)
r = lambda *s: torch.randn(*s, device="cuda", generator=gen)
X = {"q": (r(1, T, H, 128) * .3).bfloat16(), "k": (r(1, T, H, 128) * .3).bfloat16(),
     "v": (r(1, T, H, 128) * .3).bfloat16(), "g": (r(1, T, H, 128) * .3).bfloat16(),
     "beta": r(1, T, H).sigmoid(), "A_log": torch.log(1 + 15 * torch.rand(H, device="cuda", generator=gen)),
     "dt_bias": r(H * 128) * .1}
L = {k: v.requires_grad_(True) for k, v in X.items()}
D_O = r(1, T, H, 128).bfloat16(); CU = torch.tensor([0, T], dtype=torch.int32, device="cuda")
kw = dict(A_log=L["A_log"], dt_bias=L["dt_bias"], output_final_state=False, use_qk_l2norm_in_kernel=True,
          use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0, cu_seqlens=CU, cu_seqlens_cpu=CU.cpu())
def step(fn):
    q, k, v, g = (L[n] * 1.0 for n in "qkvg")
    o = fn(q, k, v, g, L["beta"], **kw)[0]
    del q, k, v, g
    o.backward(D_O)
ENV = os.environ.get("FLASH_TRAIN_KDA_SAVE_QKN", "-")
for name, fn in (("g2", g2), ("ftc", ft.chunk_kda)):
    for _ in range(2):
        step(fn)
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated(); step(fn); torch.cuda.synchronize()
    print(f"{name} T={T} H={H} env={ENV} peak_over_inputs_GiB={(torch.cuda.max_memory_allocated() - base) / 2**30:.3f}")
