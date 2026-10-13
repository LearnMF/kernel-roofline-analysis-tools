"""(b) precision study: emulate a narrower handoff of the backward's fp32 intermediates with the REAL
kernels.  KDA_G2_HQ="dqkg:bf16,daqk:bf16" (formats bf16 | fp16) rounds the named tensor once (RNE)
at its producer->consumer boundary -- identical to the producer storing the narrow type and the
consumer widening it back to fp32:
  dqkg  dq, dk, dg (fp32) between wy_s4_split (s4b) and bwd_tail        1.5 KB/token*head
  daqk  dAqk (fp32) between dav and wy_s4_split (s4b)                  0.25 KB/token*head
python3 hq_emulate.py <tree>   (patches hip_kda/ops/k3_g2.py; no rebuild needed)"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "hip_kda" / "ops" / "k3_g2.py"
t = P.read_text()


def rep(old, new):
    global t
    assert t.count(old) == 1, (old[:80], t.count(old))
    t = t.replace(old, new)


rep("""def _thd4(x):""", '''def _hq(name, *xs):
    """kra (b) precision study: KDA_G2_HQ="dqkg:bf16,daqk:fp16" rounds a handoff once (RNE)."""
    spec = dict(p.split(":") for p in os.environ.get("KDA_G2_HQ", "").split(",") if ":" in p)
    fmt = spec.get(name)
    if fmt is None:
        return xs if len(xs) > 1 else xs[0]
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16}[fmt]
    out = tuple(x.to(dt).to(x.dtype) for x in xs)
    return out if len(out) > 1 else out[0]


def _thd4(x):''')
rep("""        if no_ando:
            aN_do = torch.empty(0, dtype=do.dtype, device=do.device)
        del Vn""", """        if no_ando:
            aN_do = torch.empty(0, dtype=do.dtype, device=do.device)
        dAqk = _hq("daqk", dAqk)
        del Vn""")
rep("""        if hipi:
            # F1: every contract on the HIP tail""", """        dq, dk, dg = _hq("dqkg", dq, dk, dg)
        if hipi:
            # F1: every contract on the HIP tail""")
P.write_text(t)
print("HQ emulation APPLIED")
