### H20 FLA default / ours (device time; > 1 = ours faster)

| T | TP | H20 FLA ms | 2026-10-10 first cross-platform run | 2026-10-12 first H20 run (a81ded8) | 2026-10-16 session start (P-65) | 2026-10-17 now (P-87, 931f9e3) |
|---|---|---:|---:|---:|---:|---:|
| 8K | TP8 | 2.64 | 0.64 (4.15) | 0.74 (3.58) | 0.77 (3.42) | 0.85 (3.11) |
| 8K | TP4 | 4.88 | 0.63 (7.80) | 0.72 (6.78) | 0.79 (6.17) | 0.88 (5.57) |
| 8K | TP2 | 9.22 | 0.60 (15.37) | 0.69 (13.41) | 0.70 (13.20) | 0.74 (12.45) |
| 8K | TP1 | 18.07 | 0.52 (35.05) | 0.68 (26.77) | 0.68 (26.55) | 0.72 (25.10) |
| 16K | TP8 | 5.15 | 0.66 (7.86) | 0.76 (6.75) | 0.81 (6.36) | 0.89 (5.79) |
| 16K | TP4 | 9.68 | -- | 0.74 (13.09) | 0.80 (12.05) | 0.89 (10.84) |
| 16K | TP2 | 18.36 | -- | 0.69 (26.43) | 0.75 (24.53) | 0.81 (22.62) |
| 16K | TP1 | 36.14 | -- | 0.67 (53.97) | 0.68 (53.46) | 0.71 (51.13) |
| 32K | TP8 | 10.23 | 0.67 (15.17) | 0.79 (12.94) | 0.84 (12.14) | 0.92 (11.12) |
| 32K | TP4 | 19.20 | -- | 0.75 (25.63) | 0.76 (25.15) | 0.82 (23.52) |
| 32K | TP2 | 36.72 | -- | 0.73 (50.13) | 0.76 (48.44) | 0.84 (43.88) |
| 32K | TP1 | 72.40 | -- | 0.70 (102.99) | 0.70 (103.65) | 0.72 (100.54) |
| 64K | TP8 | 20.26 | -- | 0.83 (24.38) | 0.89 (22.81) | 0.99 (20.43) |
| 64K | TP4 | 38.65 | -- | 0.80 (48.36) | 0.82 (47.21) | 0.90 (43.07) |
| 64K | TP2 | 73.59 | -- | 0.75 (98.37) | 0.79 (93.46) | 0.84 (87.17) |
| 64K | TP1 | 145.95 | -- | 0.67 (218.02) | 0.69 (212.01) | 0.72 (201.61) |

geomean all 16: 2026-10-10 first cross-platform run: --, 2026-10-12 first H20 run (a81ded8): 0.730, 2026-10-16 session start (P-65): 0.762, 2026-10-17 now (P-87, 931f9e3): 0.823

geomean 9 priority (16K-64K, TP>1): 2026-10-10 first cross-platform run: --, 2026-10-12 first H20 run (a81ded8): 0.760, 2026-10-16 session start (P-65): 0.801, 2026-10-17 now (P-87, 931f9e3): 0.876

geomean 6 shapes of 10-10: 2026-10-10 first cross-platform run: 0.616, 2026-10-12 first H20 run (a81ded8): 0.728, 2026-10-16 session start (P-65): 0.764, 2026-10-17 now (P-87, 931f9e3): 0.829

### 16K x 12 heads (TP8; device us)

| stage | ours 10-12 | ours now | FLA on H20 | H20 / ours now | ours now - H20 |
|---|---:|---:|---:|---:|---:|
| forward prep (l2norm, gate, intra A, w/u) | 963 | 1014 | 642 | 0.63 | +371 |
| forward recurrence h + o | 671 | 573 | 582 | 1.01 | -8 |
| bwd recompute (gate, w/u [+v_new], h) | 1523 | 833 | 724 | 0.87 | +109 |
| dA_qk + dv (dAv) | 280 | 280 | 86 | 0.31 | +193 |
| state gradient dh (dhu) | 675 | 660 | 438 | 0.66 | +222 |
| dq/dk/dg/dbeta/dA_kk (wy + intra) | 1894 | 1785 | 2324 | 1.30 | -539 |
| tail (cumsum, gate bwd, l2norm bwd) | 713 | 636 | 256 | 0.40 | +380 |
| other (copies, fills, reductions) | 20 | 20 | 108 | 5.47 | -88 |
| **total** | **6740** | **5801** | **5160** | **0.89** | **+640** |

### 16K x 48 heads (TP2; device us)

| stage | ours 10-12 | ours now | FLA on H20 | H20 / ours now | ours now - H20 |
|---|---:|---:|---:|---:|---:|
| forward prep (l2norm, gate, intra A, w/u) | 3757 | 3728 | 2511 | 0.67 | +1216 |
| forward recurrence h + o | 1793 | 2397 | 1486 | 0.62 | +912 |
| bwd recompute (gate, w/u [+v_new], h) | 5572 | 3370 | 2043 | 0.61 | +1327 |
| dA_qk + dv (dAv) | 1089 | 1090 | 330 | 0.30 | +761 |
| state gradient dh (dhu) | 2465 | 2017 | 1304 | 0.65 | +713 |
| dq/dk/dg/dbeta/dA_kk (wy + intra) | 8940 | 7474 | 9328 | 1.25 | -1853 |
| tail (cumsum, gate bwd, l2norm bwd) | 2754 | 2501 | 999 | 0.40 | +1502 |
| other (copies, fills, reductions) | 46 | 44 | 326 | 7.38 | -282 |
| **total** | **26417** | **22622** | **18326** | **0.81** | **+4296** |

### 32K x 24 heads (TP4; device us)

| stage | ours 10-12 | ours now | FLA on H20 | H20 / ours now | ours now - H20 |
|---|---:|---:|---:|---:|---:|
| forward prep (l2norm, gate, intra A, w/u) | 3720 | 4071 | 2537 | 0.62 | +1534 |
| forward recurrence h + o | 1952 | 2345 | 1785 | 0.76 | +560 |
| bwd recompute (gate, w/u [+v_new], h) | 5524 | 3334 | 2350 | 0.70 | +985 |
| dA_qk + dv (dAv) | 1091 | 1092 | 335 | 0.31 | +757 |
| state gradient dh (dhu) | 2577 | 2437 | 1598 | 0.66 | +839 |
| dq/dk/dg/dbeta/dA_kk (wy + intra) | 7984 | 7675 | 9276 | 1.21 | -1602 |
| tail (cumsum, gate bwd, l2norm bwd) | 2735 | 2515 | 1002 | 0.40 | +1513 |
| other (copies, fills, reductions) | 42 | 41 | 327 | 7.93 | -286 |
| **total** | **25626** | **23510** | **19211** | **0.82** | **+4299** |

### 64K x 48 heads (TP2; device us)

| stage | ours 10-12 | ours now | FLA on H20 | H20 / ours now | ours now - H20 |
|---|---:|---:|---:|---:|---:|
| forward prep (l2norm, gate, intra A, w/u) | 14148 | 14832 | 10242 | 0.69 | +4589 |
| forward recurrence h + o | 8968 | 7822 | 5961 | 0.76 | +1861 |
| bwd recompute (gate, w/u [+v_new], h) | 21190 | 13762 | 8192 | 0.60 | +5570 |
| dA_qk + dv (dAv) | 4013 | 3921 | 1344 | 0.34 | +2577 |
| state gradient dh (dhu) | 9380 | 8028 | 5170 | 0.64 | +2858 |
| dq/dk/dg/dbeta/dA_kk (wy + intra) | 30135 | 29737 | 37234 | 1.25 | -7496 |
| tail (cumsum, gate bwd, l2norm bwd) | 10447 | 9878 | 4082 | 0.41 | +5796 |
| other (copies, fills, reductions) | 141 | 145 | 1219 | 8.42 | -1074 |
| **total** | **98422** | **88126** | **73444** | **0.83** | **+14682** |
