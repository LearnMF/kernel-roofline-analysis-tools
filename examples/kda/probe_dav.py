"""dav-only timing probe: g2_ops.dav(do, Vn, Aqk) on random inputs of the production layouts,
median of `reps` event-timed launches (after warm-up).  Used for per-part ablation builds.
    python3 probe_dav.py T H [reps=50]          (HIP_KDA_ROOT / PYTHONPATH -> the tree)"""
import json
import os
import statistics as st
import sys

import torch

T, H = int(sys.argv[1]), int(sys.argv[2])
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 50
from hip_kda.ops import g2_ops  # noqa: E402

torch.manual_seed(20261011)
NT = T // 64
do = (torch.randn(1, T, H, 128, device="cuda") * 0.3).bfloat16()
Vn = (torch.randn(1, H, NT, 4, 8, 4, 16, 4, device="cuda") * 0.3).bfloat16()
Aqk = (torch.randn(1, T, H, 64, device="cuda") * 0.1).bfloat16()
kw = {} if os.environ.get("PROBE_ANDO", "1") == "1" else {"emit_ando": False}   # 0 = production route
for _ in range(5):
    g2_ops.dav(do, Vn, Aqk, 0.0884, H, NT, **kw)
torch.cuda.synchronize()
ts = []
for _ in range(reps):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); g2_ops.dav(do, Vn, Aqk, 0.0884, H, NT, **kw); e.record()
    torch.cuda.synchronize()
    ts.append(s.elapsed_time(e) * 1000.0)
print(json.dumps({"tag": "dav", "T": T, "H": H, "us": round(st.median(ts), 2), "min_us": round(min(ts), 2)}))
