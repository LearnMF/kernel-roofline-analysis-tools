"""gfx938 four-floor model of the whole KDA forward + backward chain (from the gfx938 ceiling cards).

  theoretical  = max(algorithm-required HBM bytes / HBM peak, algorithm MMAC FLOPs / MMAC peak)   (whole chain)
  decomposition= sum over kernels of max(measured HBM bytes / HBM peak, MMAC FLOPs / MMAC peak)     (this split)
  fully loaded = sum over kernels of max(every per-resource floor of the card: HBM, TA, VALU, LDS, MMAC, chain)
  measured     = sum of kernel times
Algorithm-required bytes per token-head (Megatron-LM-k3 call, bf16 tensors, fp32 beta):
  fwd: read q, k, v, g (4 x 256 B) + beta 4 B, write o 256 B
  bwd: read q, k, v, g, do (5 x 256 B) + beta, write dq, dk, dv, dg (4 x 256 B) + dbeta 4 B
  (+ the minimal saved set between them is not counted: it is an implementation choice).
Algorithm MMAC FLOPs: the measured kernels' MMAC FLOPs (cards 'flops'), i.e. the current algorithm's GEMM work.
    python3 floors938.py [cards.json] [machine.json]"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
C = json.load(open(sys.argv[1] if len(sys.argv) > 1 else ROOT / "results/kda-gfx938-cards-20261025/cards.json"))
MACH = json.load(open(sys.argv[2] if len(sys.argv) > 2 else ROOT / "machines/gfx938-bw1102.json"))
P = MACH["peaks"]
HBM_R, HBM_W = P["hbm_read"]["value"] * 1e9, P["hbm_write"]["value"] * 1e9
MMAC = P["mmac_bf16"]["value"] * 1e12 * 0.92
for shape, v in C.items():
    T, H = map(int, shape.split("_"))
    cards = v["cards"]
    tok = T * H
    rd = tok * (4 * 256 + 4) + tok * (5 * 256 + 4)
    wr = tok * 256 + tok * (4 * 256 + 4)
    flops = sum(c.get("flops") or 0 for c in cards.values())
    theo = max(rd / HBM_R + wr / HBM_W, flops / MMAC) * 1e6
    dec = sum(max((c.get("hbm_rd") or 0) / HBM_R + (c.get("hbm_wr") or 0) / HBM_W, (c.get("flops") or 0) / MMAC) * 1e6 for c in cards.values())
    full = sum(c["floor_us"] for c in cards.values())
    meas = sum(c["t_us"] for c in cards.values())
    hbm_meas = sum((c.get("hbm_rd") or 0) + (c.get("hbm_wr") or 0) for c in cards.values())
    print(f"== {T}/{H} (gfx938): theoretical {theo:8.1f} us | decomposition {dec:8.1f} | fully loaded {full:8.1f} | measured {meas:8.1f}"
          f"   ratios meas/theo {meas/theo:5.2f}  meas/full {meas/full:4.2f}  HBM moved {hbm_meas/1e6:.0f} MB vs required {(rd+wr)/1e6:.0f} MB ({hbm_meas/(rd+wr):.1f}x)")
    bind = {}
    for k, c in cards.items():
        bind.setdefault(c.get("primary") or c.get("binding"), []).append((k, c["t_us"]))
    print("   time by primary resource:", {b: round(sum(t for _, t in ks), 1) for b, ks in bind.items()})
    top = sorted(cards.items(), key=lambda kv: -kv[1]["t_us"])[:6]
    for k, c in top:
        hb = ((c.get("hbm_rd") or 0) / HBM_R + (c.get("hbm_wr") or 0) / HBM_W) * 1e6
        print(f"   {k:18s} t {c['t_us']:8.1f}  HBM-floor {hb:7.1f}  MMAC {((c.get('flops') or 0)/MMAC*1e6):6.1f}  TA {c.get('f_ta',0):7.1f}  VALU {c.get('f_val',0):7.1f}  LDS {c.get('f_lds',0):7.1f}  primary {c.get('primary')}")
