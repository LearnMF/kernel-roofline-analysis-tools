# KDA gfx938: 5ebce6f..2631309 push summary (local branch feat/kda-gfx938-mk3, not pushed)

Branch policy: one source tree for gfx936 and gfx938 (arch-specific paths behind `__gfx938__` / runtime arch checks),
merged into the mainline; feat/kda-gfx938-mk3 is a development branch only.  Every change is gated on both archs
(bitwise gate + timing).

## Commits (all gfx938-only code paths; gfx936 unchanged, bit-identical numerics)
| commit | id | change | kernel effect (9-shape ABBA vs parent) |
|---|---|---|---|
| 859a229 | P-938pk | bwd_tail Kahan scan as packed fp32 | bwd_tail 1.007-1.014x |
| 66fadb3 | P-938pk2 | bwd_tail phase A as packed fp32 pairs; a - b as v_pk_fma(b, -1, a) | bwd_tail 1.085-1.220x |
| 24b9d4c | P-938pk2 | prep_a gate scan / carry / QgA / Kb / Kn / P1 exp2 arguments packed | prep_a 1.033-1.062x |
| 69de6cf | P-938va | dav re-emits v_new in A form in place; s4b skips its C -> A transpose | s4b 1.008-1.056x, dav 0.865-0.943x |
| 7e4601b | P-938a16 | prep_a in a 16 KB arena -> 4 CTAs per CU | prep_a 1.005-1.279x (1.25-1.28x at 256 CTAs) |

Dev knobs (FTRAIN_KDA_DEV only): KDA_G2_VNA=0 (P-938va off).  No retained-memory change.

## End-to-end (24 shapes, mk3_fb.py, 8 fresh processes per shape in ABBA x 2, nmz5)
- 7e4601b vs 5ebce6f: **geomean fwd 1.022, bwd 1.020, total 1.021 (range 1.000-1.051), no shape slower**
  (doc §131; data results/kda-gfx938-24shape-20261103/).
- vs H20 FLA default (separate 11-02 H20 run, session-dependent): total ~1.00-1.03, forward ~0.90-0.94,
  backward ~1.04-1.06.  The remaining deficit is the forward.

## Verification (7e4601b)
| check | result |
|---|---|
| bitwise gate vs 5ebce6f, gfx938, default route (84 tensors incl. varlen / l2norm / gate) | identical |
| same, keep-state route (FTRAIN_KDA_KEEP_STATE=1, SAVE_H=0) | identical |
| same, recompute route (KEEP_STATE=0, SAVE_H=0) | identical |
| bitwise gate vs 5ebce6f on gfx936 (bw61) | identical |
| pytest tests/python (gfx938) | 104 passed |
| ftrain_test (ctest binary) gfx936 | 181 / 181 passed |
| ftrain_test gfx938 | 181 / 181 after 2631309 (the API test expected UNSUPPORTED on every non-gfx936 device -- a gfx938-port leftover, fixed in its own commit) |
| gfx936 performance, 7e4601b vs 5ebce6f (bw61, kbench ABBA, 9 shapes) | fwd+bwd 0.999-1.016x; s4b 0.999-1.039x, dav 0.974-1.007x (the gfx936 code paths of the two kernels changed only by an inactive runtime branch / emit block) -- no regression |
| resources | no scratch in production instances; VGPR / LDS per commit message |

Note: P-938va is bit-identical by construction, so the keep-state gate shows the route is numerically unchanged;
that dav skips the A-form write-back there follows from the runner condition (vna = !kept && gfx938).

## Findings recorded (docs/kda_gfx938_round.md §112-138)
- Packed fp32 works on gfx938 when subtraction is written as fma(b, -1, a) (supersedes §54 / §106).
- Residency (R70): ~768 VGPRs per lane per SIMD -- 168 VGPRs allow 4 CTAs per CU, 200-256 allow 3; LDS <= 16 KB for 4.
- R69: lane-permuted 16 B global loads within 16-lane groups cost ~25% on recurrence kernels.
- Negative / closed: A1 (s4a / dav addresses), bwd_tail 3-wave, Kb chunk-native layout, B1 (dhu double buffer),
  B2 (dhu lit1 lts1), C2 (s4b row-slice prefetch), s4b 3rd CTA (P-938s1 bisect, dwT prefetch), s4a 16 KB (VGPR-bound),
  prep fusion on 16 KB (P-938ab, 0.83-0.86x), s4b bf16 tables (fails FP64 bar 1.75-3.7x), chunk size 32 / 128.
- FLA's chunk algorithm is the same family as ours (factorized decay, sub-block references, blocked inverse); the
  forward gap to H20 is hardware throughput (VALU / SFU 1.65-3.3x).
