"""Wave-quantization probe for the per-head recurrences (each CTA runs the WHOLE sequence, so CTAs
beyond the resident slots run as a later, latency-bound round).  Times g2_ops.fwd_h_bn (v8o4,
4 V-slab CTAs per head) and g2_ops.dhu_bn on random native operands for a list of head counts at
fixed T: a step in time(H) where 4H crosses the resident-CTA capacity confirms the tail round.
    python3 probe_wave_quant.py T H1,H2,...   (HIP_KDA_ROOT/PYTHONPATH -> the tree)"""
import json
import statistics as st
import sys

import torch

from hip_kda.ops import g2_ops

T = int(sys.argv[1])
Hs = [int(x) for x in sys.argv[2].split(",")]
NT = T // 64
torch.manual_seed(7)


def timed(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    return round(st.median(ts), 1)


for H in Hs:
    r = lambda *shape: (torch.randn(*shape, device="cuda") * 0.05).bfloat16()
    Wn, Kn, Un = r(1, H, NT, 4, 8, 4, 16, 4), r(1, H, NT, 8, 4, 4, 16, 4), r(1, H, NT, 4, 8, 4, 16, 4)
    Gn = -(torch.rand(1, H, NT, 8, 16, device="cuda") * 0.1)
    KgA, Qn, Wb = r(1, H, NT, 4, 8, 4, 16, 4), r(1, H, NT, 8, 4, 4, 16, 4), r(1, H, NT, 8, 4, 4, 16, 4)
    dOT, dVn = r(1, H, NT, 8, 4, 4, 16, 4), r(1, H, NT, 4, 8, 4, 16, 4)
    QgA, AqkA = r(1, H, NT, 4, 8, 4, 16, 4), r(1, H, NT, 4, 4, 4, 16, 4)
    t_fo = timed(lambda: g2_ops.fwd_h_o(Wn, Kn, Un, Gn, QgA, AqkA, 0.0884, o_bf16=True, emit_vn=False))
    t_bn = timed(lambda: g2_ops.fwd_h_bn(Wn, Kn, Un, Gn, emit_an=False))
    t_dhu = timed(lambda: g2_ops.dhu_bn(KgA, Qn, Wb, dOT, dVn, Gn, 0.0884, emit_dv2n=False))
    print(json.dumps({"T": T, "H": H, "ctas_v8o4": 4 * H, "fwd_h_o_us": t_fo, "fwd_h_bn_us": t_bn, "dhu_us": t_dhu,
                      "fwd_h_bn_us_per_head": round(t_bn / H, 2), "dhu_us_per_head": round(t_dhu / H, 2)}), flush=True)
    del Wn, Kn, Un, Gn, KgA, Qn, Wb, dOT, dVn, QgA, AqkA
