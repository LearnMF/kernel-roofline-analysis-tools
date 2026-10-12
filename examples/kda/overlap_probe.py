"""Can gfx936 overlap the latency-bound forward recurrence (fwd_h_o, H x 4 CTAs) with the parallel,
VALU-heavy prep (fwd_prep) of independent data?  serial vs two-stream concurrent wall time."""
import statistics as st
import sys

import torch

from hip_kda.ops import g2_ops

T, H = int(sys.argv[1]), int(sys.argv[2])
NT = T // 64
dev = "cuda"
torch.manual_seed(0)


def inputs():
    q = torch.randn(1, T, H, 128, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, T, H, 128, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, T, H, 128, device=dev, dtype=torch.bfloat16)
    g = torch.randn(1, T, H, 128, device=dev, dtype=torch.bfloat16)
    beta = torch.rand(1, T, H, device=dev, dtype=torch.float32)
    a_log = torch.randn(H, device=dev, dtype=torch.float32) * 0.1
    dt = torch.randn(H * 128, device=dev, dtype=torch.float32) * 0.1
    return q, k, v, g, beta, a_log, dt


RCP_LN2 = 1.4426950408889634
scale = 128 ** -0.5
prep = lambda x: g2_ops.fwd_prep(x[0], x[1], x[2], x[3], x[5], x[6], x[4], -5.0, RCP_LN2, True, scale, H, NT)
s1 = inputs(); s2 = inputs()
nat = prep(s1)                     # natives for the recurrence (set 1)
Wn, Kn, Un, QgA, AqkA, Aqk, Akk, qr, kr, Gn = nat
rec = lambda: g2_ops.fwd_h_o(Wn, Kn, Un, Gn, QgA, AqkA, scale, o_bf16=True, emit_vn=False)
pre = lambda: prep(s2)
sa, sb = torch.cuda.Stream(), torch.cuda.Stream()


def timed(fn, n=15):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        e0.record(); fn(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return st.median(ts)


def conc():
    cur = torch.cuda.current_stream()
    sa.wait_stream(cur); sb.wait_stream(cur)
    with torch.cuda.stream(sa):
        rec()
    with torch.cuda.stream(sb):
        pre()
    cur.wait_stream(sa); cur.wait_stream(sb)


t_rec = timed(rec); t_pre = timed(pre)
t_ser = timed(lambda: (pre(), rec())); t_con = timed(conc)
print(f"T={T} H={H}: recurrence {t_rec:.3f} ms, prep {t_pre:.3f} ms, serial {t_ser:.3f} ms, "
      f"two-stream {t_con:.3f} ms -> overlap saves {t_ser - t_con:.3f} ms "
      f"({(t_ser - t_con) / min(t_rec, t_pre):.0%} of the shorter one)")
