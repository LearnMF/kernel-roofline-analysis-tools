"""(T, H) aliasing probe: per-kernel backward time with the inputs at their natural (allocator)
addresses vs staggered by i * PAD bytes inside one buffer.  python3 mk3_align.py T H PAD_MiB"""
import json, sys
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
import flash_train.torch as ft
T, H, PAD = int(sys.argv[1]), int(sys.argv[2]), int(float(sys.argv[3]) * 2**20)
gen = torch.Generator(device="cuda").manual_seed(1)
n = T * H * 128
def place(vals):
    if PAD == 0:
        return [v.clone() for v in vals]
    tot = sum(v.numel() * v.element_size() + PAD for v in vals) + 2**20
    buf = torch.empty(tot, dtype=torch.uint8, device="cuda"); out, off = [], 0
    for i, v in enumerate(vals):
        off += PAD
        b = v.numel() * v.element_size()
        t = buf[off:off + b].view(v.dtype).view(v.shape); t.copy_(v); out.append(t); off += b
    return out
r = lambda *s: torch.randn(*s, device="cuda", generator=gen)
q, k, v, g, do = place([F.normalize(r(1, T, H, 128), dim=-1).bfloat16(), F.normalize(r(1, T, H, 128), dim=-1).bfloat16(),
                        (r(1, T, H, 128) * .3).bfloat16(), (r(1, T, H, 128) * .3).bfloat16(), r(1, T, H, 128).bfloat16()])
L = dict(q=q, k=k, v=v, g=g, beta=r(1, T, H).sigmoid(), A_log=torch.log(1 + 15 * torch.rand(H, device="cuda", generator=gen)), dt_bias=r(H * 128) * .1)
for t in L.values(): t.requires_grad_(True)
f = lambda: ft.chunk_kda(L["q"], L["k"], L["v"], L["g"], L["beta"], A_log=L["A_log"], dt_bias=L["dt_bias"], use_qk_l2norm_in_kernel=False,
                         use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0)[0]
for _ in range(3):
    for t in L.values(): t.grad = None
    f().backward(do)
torch.cuda.synchronize()
o = f(); N = 5
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for _ in range(N):
        for t in L.values(): t.grad = None
        o.backward(do, retain_graph=True)
    torch.cuda.synchronize()
kt = {}
for e in p.key_averages():
    if e.device_type.name == "CUDA" and e.device_time_total > 0:
        kt[e.key] = kt.get(e.key, 0) + e.device_time_total / 1e3 / N
gs = lambda s: round(sum(v for kk, v in kt.items() if s in kk), 3)
print(json.dumps({"T": T, "H": H, "pad_MiB": PAD / 2**20, "bwd": round(sum(kt.values()), 3), "s4b": gs("p3_wy_s4b"), "s4a": gs("p3_wy_s4a"),
                  "wu": gs("g2_wu"), "dhu": gs("dhu"), "tail": gs("bwd_tail"), "addr_mod_32MiB": [(t.data_ptr() % (32 << 20)) >> 20 for t in (q, k, v, g, do)]}))
