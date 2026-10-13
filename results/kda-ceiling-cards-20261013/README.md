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
- `sqtt_attribution.md` — SQTT stall attribution for 16K/H12 and 16K/H48,
  **captured on the production driver** (`kda_run.py --tree g2_card_p52` under
  xprof `--enable-sqtt`, static binary copied into the prod container; exported
  via XCompute CLI `hot_inst`/`bubble_summary`). All 14 production kernels with
  their true template instances (fwd_h MODE1/MODE4, dhu<1,true>, prep_a<1,false>).
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


## 修订记录（docs/tasks/kda_ceiling_cards_fix.md）

沿用不重测：16 形态 trace×5 + 4 组 PMC + optrace、ISA 统计、136 项核对、标定复核。
新测量：生产驱动 SQTT（16K/12、16K/48）、P-53+P-54a 增量（16K×4 形态，
`raw/p54_16384_*/`，冻结副本 `/opt/kda_env/g2_card_p54`）。

| 修订 | 内容 |
|---|---|
| F1 | SQTT 改用生产驱动 kda_run.py；符号核对见 verification.md F1 |
| F2 | "下限"统一改名"现指令流下限"；新增"接口下限"（必需字节×带宽 + 理想TA 21/56cyc/KB + 算力/链） |
| F3 | 图C/表3 主指标改为 TA 放大；VMEM 指令倍数降为参考列 |
| F4 | 表4 新增消融/实测证据列；冲突类"病灶"判断移除（R42/R43 消融 0–2.5%） |
| F5 | cards.md 新增增量节：4 形态 fwd_h#1/bwd_tail 前后对比 + baseline-shift 校正 |
| F6 | 图A 右栏 8 通道利用率条 + 0.85/0.6 参考线 + 算子方块点；fig_op_gap.png；图D 改版；fig_gap_matrix.png |
| F7 | 结论修订：流下限≈实测只说明现有指令流已接近满速；余量在 TA 放大（1.3–5.5×）与接口下限缺口 |

## 图名对照（任务 §7.3 → 文件）

| 任务名 | 文件 | 内容 |
|---|---|---|
| 图 A（每形态） | `figA_<T>_<H>.png` ×16 | 左：分层 roofline（LDS◆/L2▼/HBM● + 算子□）；右：每 kernel 8 通道利用率条 |
| 图 B | `figB.png` | 每 kernel 现指令流下限 vs 实测 |
| 图 C | `figC.png` | TA 放大（主）+ w_gap 字节倍率 |
| 图 D | `figD.png` | 时间占比热力图（格子=占比+主约束.利用率；边框=binding） |
| 算子缺口图 | `fig_op_gap.png` | 16 形态 ×（实测堆叠/Σ流下限/Σ接口下限/第一性下限） |
| 缺口矩阵 | `fig_gap_matrix.png` | kernel×形态：缺口类型 + 相对接口下限的潜在收益 |
