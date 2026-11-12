import sys
import torch

mode = sys.argv[1]
if mode == "fla_first":
    import fla_prune_ns4
    fla_prune_ns4.apply(verbose=False)
    from fla.ops.kda import chunk_kda as _fk  # noqa: F401
    print("fla imported", flush=True)
from hip_kda.fla import chunk_kda

T, H = 8192, 12
dev = "cuda"
gen = torch.Generator(device="cuda").manual_seed(T + H)
r = lambda *s: torch.randn(*s, device="cuda", generator=gen)
X = {"q": (r(1, T, H, 128) * .3).bfloat16(), "k": (r(1, T, H, 128) * .3).bfloat16(),
     "v": (r(1, T, H, 128) * .3).bfloat16(), "g": (r(1, T, H, 128) * .3).bfloat16(),
     "beta": r(1, T, H).sigmoid(), "A_log": torch.log(1 + 15 * torch.rand(H, device="cuda", generator=gen)),
     "dt_bias": r(H * 128) * .1}
D_O = r(1, T, H, 128).bfloat16()
CU = torch.tensor([0, T], dtype=torch.int32, device="cuda")
LEAVES = {k: v.requires_grad_(True) for k, v in X.items()}
print("inputs ok", flush=True)
L = LEAVES
kw = dict(A_log=L["A_log"], dt_bias=L["dt_bias"], scale=None, output_final_state=False,
          use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=False,
          safe_gate=True, lower_bound=-5.0, cu_seqlens=CU, cu_seqlens_cpu=CU.cpu())
o = chunk_kda(L["q"], L["k"], L["v"], L["g"], L["beta"], **kw)[0]
torch.cuda.synchronize()
print("fwd ok", flush=True)
torch.autograd.grad(o, list(LEAVES.values()), D_O)
torch.cuda.synchronize()
print("bwd ok", flush=True)
