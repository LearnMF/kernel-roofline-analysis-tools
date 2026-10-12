
### 16K x 12 heads (TP8; device us; >1 = ours faster)

| stage | ours (BW) | FLA (H20) | H20 / ours | H20 x hw / ours |
|---|---:|---:|---:|---:|
| forward: prep (l2norm, gate, intra A, w/u) | 1138 | 642 | 0.56 | 1.63 |
| forward: recurrence h + output o | 671 | 582 | 0.87 | 2.51 |
| bwd recompute: q/k norm, gate, w/u, h | 1348 | 724 | 0.54 | 1.55 |
| bwd: dA_qk + dv (dAv) | 280 | 86 | 0.31 | 0.89 |
| bwd: state gradient dh (dhu) | 675 | 438 | 0.65 | 1.88 |
| bwd: dq/dk/dg/dbeta/dA_kk (wy + intra) | 1894 | 2324 | 1.23 | 3.55 |
| bwd tail: cumsum, gate bwd, l2norm bwd | 713 | 256 | 0.36 | 1.04 |
| other (copies, fills, reductions) | 20 | 108 | 5.30 | 15.32 |
| **total** | **6740** | **5160** | **0.77** | **2.21** |

### 16K x 48 heads (TP2; device us; >1 = ours faster)

| stage | ours (BW) | FLA (H20) | H20 / ours | H20 x hw / ours |
|---|---:|---:|---:|---:|
| forward: prep (l2norm, gate, intra A, w/u) | 4429 | 2511 | 0.57 | 1.64 |
| forward: recurrence h + output o | 1793 | 1486 | 0.83 | 2.39 |
| bwd recompute: q/k norm, gate, w/u, h | 4900 | 2043 | 0.42 | 1.20 |
| bwd: dA_qk + dv (dAv) | 1089 | 330 | 0.30 | 0.88 |
| bwd: state gradient dh (dhu) | 2465 | 1304 | 0.53 | 1.53 |
| bwd: dq/dk/dg/dbeta/dA_kk (wy + intra) | 8940 | 9328 | 1.04 | 3.02 |
| bwd tail: cumsum, gate bwd, l2norm bwd | 2754 | 999 | 0.36 | 1.05 |
| other (copies, fills, reductions) | 46 | 326 | 7.12 | 20.59 |
| **total** | **26417** | **18326** | **0.69** | **2.00** |

### 32K x 24 heads (TP4; device us; >1 = ours faster)

| stage | ours (BW) | FLA (H20) | H20 / ours | H20 x hw / ours |
|---|---:|---:|---:|---:|
| forward: prep (l2norm, gate, intra A, w/u) | 4393 | 2537 | 0.58 | 1.67 |
| forward: recurrence h + output o | 1952 | 1785 | 0.91 | 2.64 |
| bwd recompute: q/k norm, gate, w/u, h | 4852 | 2350 | 0.48 | 1.40 |
| bwd: dA_qk + dv (dAv) | 1091 | 335 | 0.31 | 0.89 |
| bwd: state gradient dh (dhu) | 2577 | 1598 | 0.62 | 1.79 |
| bwd: dq/dk/dg/dbeta/dA_kk (wy + intra) | 7984 | 9276 | 1.16 | 3.36 |
| bwd tail: cumsum, gate bwd, l2norm bwd | 2735 | 1002 | 0.37 | 1.06 |
| other (copies, fills, reductions) | 42 | 327 | 7.78 | 22.49 |
| **total** | **25626** | **19211** | **0.75** | **2.17** |

### 64K x 48 heads (TP2; device us; >1 = ours faster)

| stage | ours (BW) | FLA (H20) | H20 / ours | H20 x hw / ours |
|---|---:|---:|---:|---:|
| forward: prep (l2norm, gate, intra A, w/u) | 16809 | 10242 | 0.61 | 1.76 |
| forward: recurrence h + output o | 8968 | 5961 | 0.66 | 1.92 |
| bwd recompute: q/k norm, gate, w/u, h | 18528 | 8192 | 0.44 | 1.28 |
| bwd: dA_qk + dv (dAv) | 4013 | 1344 | 0.33 | 0.97 |
| bwd: state gradient dh (dhu) | 9380 | 5170 | 0.55 | 1.59 |
| bwd: dq/dk/dg/dbeta/dA_kk (wy + intra) | 30135 | 37234 | 1.24 | 3.57 |
| bwd tail: cumsum, gate bwd, l2norm bwd | 10447 | 4082 | 0.39 | 1.13 |
| other (copies, fills, reductions) | 141 | 1219 | 8.65 | 24.98 |
| **total** | **98422** | **73444** | **0.75** | **2.16** |
