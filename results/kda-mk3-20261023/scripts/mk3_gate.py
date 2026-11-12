"""Bitwise A/B gate for flash-train KDA (one route per process; compare the hash files).
  python3 mk3_gate.py OUT.json   -> sha256 of o and every gradient for each case
Cases cover the Megatron-LM-k3 call (pre-normalized q/k, use_qk_l2norm_in_kernel=False) with the
in-kernel gate and the precomputed decay, dense and packed (odd lengths), kp (H <= 20) and non-kp heads,
plus the K3 call (in-kernel l2norm) as a control that must not change."""
import hashlib, json, sys
import torch
import torch.nn.functional as F
import flash_train.torch as ft

CASES = [  # (name, T, H, cu_seqlens or None, l2norm_in_kernel, gate_in_kernel)
    ("mk3_d_1024_8", 1024, 8, None, False, True),
    ("mk3_d_2048_64", 2048, 64, None, False, True),
    ("mk3_d_4096_16", 4096, 16, None, False, True),
    ("mk3_d_8192_32", 8192, 32, None, False, True),
    ("mk3_d_16384_8", 16384, 8, None, False, True),
    ("mk3_vl_5000_16", 5000, 16, [0, 1000, 3333, 5000], False, True),
    ("mk3_vl_9001_40", 9001, 40, [0, 77, 4100, 9001], False, True),
    ("mk3_dg_4096_24", 4096, 24, None, False, False),
    ("mk3_vlg_3001_12", 3001, 12, [0, 1500, 3001], False, False),
    ("k3_d_8192_48", 8192, 48, None, True, True),
    ("k3_vl_5000_16", 5000, 16, [0, 1000, 3333, 5000], True, True),
]


def sha(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


res = {}
for name, T, H, cu, l2, gate in CASES:
    gen = torch.Generator(device="cuda").manual_seed(T * 31 + H)
    r = lambda *s: torch.randn(*s, device="cuda", generator=gen)
    q, k = r(1, T, H, 128), r(1, T, H, 128)
    if not l2:
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    X = {"q": q.bfloat16(), "k": k.bfloat16(), "v": (r(1, T, H, 128) * .3).bfloat16(), "beta": r(1, T, H).sigmoid()}
    if gate:
        X.update(g=(r(1, T, H, 128) * .3).bfloat16(), A_log=torch.log(1 + 15 * torch.rand(H, device="cuda", generator=gen)),
                 dt_bias=r(H * 128) * .1)
    else:
        X.update(g=-(torch.rand(1, T, H, 128, device="cuda", generator=gen) * 0.3))
    L = {n: t.requires_grad_(True) for n, t in X.items()}
    CU = None if cu is None else torch.tensor(cu, dtype=torch.int32, device="cuda")
    kw = dict(use_qk_l2norm_in_kernel=l2, cu_seqlens=CU)
    if gate:
        kw.update(A_log=L["A_log"], dt_bias=L["dt_bias"], use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0)
    o = ft.chunk_kda(L["q"], L["k"], L["v"], L["g"], L["beta"], **kw)[0]
    o.backward(r(1, T, H, 128).bfloat16())
    torch.cuda.synchronize()
    res[name] = {"o": sha(o), **{"d" + n: sha(t.grad) for n, t in L.items()}}
json.dump(res, open(sys.argv[1], "w"), indent=1)
print("cases", len(res))
