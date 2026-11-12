# Constant ledger: every number the predictor uses, and where it came from

The tool is being written **after** the KDA optimisation it is first checked against, so any constant
lifted from that result is a generalisation risk: it will not exist for the next operator or the next
card.  This ledger exists so that risk is visible instead of buried.  Rule: a constant is allowed only
if it can be re-measured by `kra calibrate` on a new card **without** optimising an operator first.

## Machine constants (per card, from `kra calibrate`)

| constant | machine file key | how it is obtained | operator-specific? |
|---|---|---|---|
| peak / attainable bf16, f16, tf32, fp32, i8 MMAC | `peaks.mmac_*` | L0 microbenchmark | no |
| HBM read / write / copy, L2, LDS bandwidth | `peaks.hbm_*`, `peaks.l2_read`, `peaks.lds_read` | L0 microbenchmark | no |
| VALU / SFU rates | `peaks.valu_fma_f32`, `peaks.sfu_exp2` | L0 microbenchmark | no |
| dispatch cost | `launch.launch_stream.best_us` | L0 microbenchmark | no |
| CU count, clock, LDS/CU, registers | `device.*` | driver query | no |

## Structural constants (per card, measured, but by a KDA-shaped microbenchmark)

| constant | value on gfx936 | source | risk |
|---|---|---|---|
| `structural.mmac_tile_eff` | 0.92 | `bound_microbench.hip`: attainable fraction of the MMAC peak with one operand register-resident | **medium** -- a tile-pattern property, not a KDA property, but measured with KDA's tile shape |
| `structural.vmem_cycles_per_kb` | read 21, write 56 | `ta_microbench.hip`: per-CU cycles for a 1 KB row-contiguous 16 B/lane access | low -- pure machine property |
| `structural.recurrence_step_us` | 1.241 / 1.015 (1 CTA/CU) | one dependent KDA chunk step, measured | **high -- KDA-specific.** Another recurrent operator needs its own step latency; a card without it gets a bound that says the serial chain is excluded, never a guessed number |

gfx938's machine file has no `structural` block at all.  The consequence is visible in its reports: no
MMAC tile efficiency (the peak itself is used) and no serial chain in the bound, stated as a note.

## Operator constants (per operator, from the algorithm)

Required FLOPs, required HBM bytes, lane-ops and transcendentals per op, the dataflow (which op reads
which tensor), the chunk size, the parallel structure of each op, and the on-chip tile rules of
`kra/model/capacity.py` are all **derived from the algorithm**, not fitted.  Two checks keep that honest:

* `kra/ops/kda_ir.py validate` compares the production decomposition's interface bytes against the
  optrace interface records of 16 shapes -- byte-exact, and it is the IR that is checked, not a constant;
* the analytic working set is compared against the measured production footprints in
  `docs/sim/m2b-split-search.md` and the deviations are reported, not corrected.

## Explicitly NOT used as prediction inputs

| thing | why not |
|---|---|
| measured production LDS/VGPR per kernel (`examples/kda/kda_partition.py` `CORE`) | that is the footprint of the kernel that was written after the optimisation; using it to search splits is fitting the answer. Validation only |
| rules R1-R66 and `thresholds.json` (`hcu_calibrated: true`) | gfx936 + KDA thresholds for **post-hoc attribution** from a trace. Using them as a priori prediction constants would bake one operator's conclusions into every prediction |
| the KDA report's 4.41 ms / 5.32 ms (`T_full`, `T_meas`) | they need the real kernels' instruction and byte counts; the predictor stops at `T_theory` / `T_split` |
| the 0.15 / 0.40 / 0.75 / 1.00 tiered GEMM efficiency of hcu-train-simulator | reverse-engineered from BW1000/BW1100 measurements; it is the thing this tool is meant to replace, not an input |
| the 530 us benchmark floor found in the BW1000 profile | an artefact of that harness's timing scope, not a hardware cost. Kept in the M1 validation report as a fitted constant of that dataset, with that label |

## Fitted constants that do exist (and are labelled as such)

| constant | where | why it is allowed |
|---|---|---|
| `eta` and the floor of the M1 留一法 comparison (`docs/sim/m1-bw1000-validation.md`) | validation only | they are fitted *on purpose*, to compare against the tiered coefficients on equal terms; they are not used by `kra predict` |
