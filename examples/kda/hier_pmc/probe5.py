import sys
import torch

import flash_attn.utils.sparse_utils  # noqa: F401  (order fix)

mode = sys.argv[1]
print("mode", mode, flush=True)
if mode == "triton_only":
    import triton
elif mode == "fla_kda_only":
    import fla.ops.kda  # noqa: F401
elif mode == "fla_kda_chunk":
    from fla.ops.kda import chunk_kda as _fk  # noqa: F401
elif mode == "prune_noapply":
    import fla_prune_ns4  # noqa: F401
elif mode == "prune_apply":
    import fla_prune_ns4
    fla_prune_ns4.apply(verbose=False)
elif mode == "plain":
    pass

from hip_kda.fla import chunk_kda

T, H = 8192, 12
dev = "cuda"
gen = torch.Generator(device="cuda").manual_seed(1)
r = lambda *s: torch.randn(*s, device="cuda", generator=gen)
q = (r(1, T, H, 128) * .3).bfloat16()
k = (r(1, T, H, 128) * .3).bfloat16()
v = (r(1, T, H, 128) * .3).bfloat16()
g = (r(1, T, H, 128) * .3).bfloat16()
beta = r(1, T, H).sigmoid()
A_log = torch.log(1 + 15 * torch.rand(H, device="cuda", generator=gen))
dt_bias = r(H * 128) * .1
CU = torch.tensor([0, T], dtype=torch.int32, device="cuda")
print("fwd", flush=True)
o = chunk_kda(q, k, v, g, beta, A_log=A_log, dt_bias=dt_bias, scale=None, output_final_state=False,
              use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=False,
              safe_gate=True, lower_bound=-5.0, cu_seqlens=CU, cu_seqlens_cpu=CU.cpu())[0]
torch.cuda.synchronize()
print("OK", flush=True)
