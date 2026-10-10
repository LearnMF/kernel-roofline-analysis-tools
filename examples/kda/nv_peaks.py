"""Attainable HBM bandwidth and bf16 matmul throughput on an NVIDIA GPU with plain torch
(the NV counterpart of the L0 numbers kra calibrate measures on HCU), best of several runs."""
import json

import torch


def best_ms(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return min(ts)


n = 1 << 29                                    # 2 GiB of fp32
x = torch.empty(n, device="cuda").normal_()
y = torch.empty_like(x)
copy_ms = best_ms(lambda: y.copy_(x))
read_ms = best_ms(lambda: x.sum())
a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
mm_ms = best_ms(lambda: a @ b, reps=10)
out = {"gpu": torch.cuda.get_device_name(0),
       "hbm_copy_GBps": round(2 * n * 4 / copy_ms / 1e6, 1),
       "hbm_read_GBps": round(n * 4 / read_ms / 1e6, 1),
       "bf16_mm_TFLOPS": round(2 * 8192 ** 3 / mm_ms / 1e9, 1),
       "sm_count": torch.cuda.get_device_properties(0).multi_processor_count}
print(json.dumps(out))
