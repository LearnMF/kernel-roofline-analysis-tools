# KDA G2 (P-52) ceiling cards — gfx936/bw7 — 2026-10-13

Frozen tree: `/opt/kda_env/g2_card_p52` fingerprint `35b7bd3a8862`.
Task spec: `docs/tasks/kda_ceiling_cards_task.md`.

## Contents

- `cards.json` — all 16 shapes (T=8/16/32/64K × H=12/24/48/96), per-kernel:
  timing (5-trace median over 5 procs), counters (PMC read/write + custom L2/SQ),
  hierarchical floors (mem/compute/issue/chain), utilization, work-gap ratios,
  and §5 classification (`primary`, `secondary`, `at_ceiling`, `gap_types`).
- `cards.md` — total table + per-shape tables 1–4.
- `figA_<T>_<H>.png` — per-shape hierarchical roofline (LDS/L2/HBM points per kernel).
- `figB.png` — per-kernel floor vs measured; `figC.png` — work amplification;
  `figD.png` — kernel time-share heatmap colored by binding resource.
- `sqtt_attribution.md` — SQTT stall attribution (xprof `--enable-sqtt` on the
  `bn_sqtt` bench harness -> `.perf` -> XCompute CLI `hot_inst`/`bubble_summary`),
  for 16K/H12 and 16K/H48 top kernels. NOTE: this DTK build's hipprof has no
  `--sqtt`; the task's hipprof command was replaced by the equivalent xprof path.
- `verification.md` — ±3% trace-vs-card timing, required-byte checks
  (wu/dav/bwd_tail), ISA-loop MMAC vs SQ_INSTS_MMOP (dev +0.000 on all
  136 checks incl. the 8-subtile dhu_v8b variant).
- `repeatability.md` — planned: 16K/H12 second-run comparison
  (raw in `raw/r2_16384_12`; see notes below).
- `raw/<T>_<H>/` — hipprof traces ×5, 4 PMC passes, optrace, hy-smi, ISA dump,
  `isa_mix.json`, sqtt perfs under `raw/sqtt_*`.

## Repeatability (16K/H12, independent 2nd collection)

Per-kernel median times: all ≤ 3.8% delta (fwd_h_v8o4#1 at 3.8% — it is a
48-CTA latency-bound kernel; its TA-busy count varies ±12% run to run while
VALU counts are bit-exact 0.0%). All other kernels ≤ 1.5% time delta.

## Calibration rechecks

- Shared-card (dev5, neighbor traffic): bf16 MMAC 471.0 (−1.1% vs 476.0),
  LDS 15248 (+0.1%), L2 5710 (−6.4%), HBM 1120 (−16.4%).
- **Exclusive card (dev4, `calib_excl_dev4.json`): HBM read 1336.0 GB/s
  (−0.2% vs reference 1339.2), L2 read 6135.4 GB/s (+0.6% vs 6100.8).**
  The reference values in `machines/gfx936-bw1000.json` are confirmed; the
  shared-card readings were neighbor contamination, not a lower ceiling.

## Pipeline

`examples/kda/ceiling/`: `collect.sh` (remote capture),
`parse_raw.py`, `isa_mix.py`, `cards.py` (+`classify`), `cards_md.py`,
`plot_cards.py` (figA–D), `verify.py`.
SQTT: `examples/kda/sqtt_capture.sh` + `kra/sqtt/xcompute.py`.
