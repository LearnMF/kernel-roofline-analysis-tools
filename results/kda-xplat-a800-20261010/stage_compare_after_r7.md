
### 8K x 12 heads (device us; ratio = other / ours, >1 = ours faster)

| stage | ours (BW) | FLA (BW) | FLA (A800) | FLA@BW / ours | FLA@A800 / ours | FLA@A800 x hw / ours |
|---|---:|---:|---:|---:|---:|---:|
| forward: prep (l2norm, gate, intra A, w/u) | 665 | 815 | 493 | 1.23 | 0.74 | 0.98 |
| forward: recurrence h + output o | 360 | 605 | 382 | 1.68 | 1.06 | 1.40 |
| bwd recompute: q/k norm, gate, w/u, h | 759 | 809 | 476 | 1.06 | 0.63 | 0.83 |
| bwd: dA_qk + dv (dAv) | 167 | 139 | 77 | 0.83 | 0.46 | 0.61 |
| bwd: state gradient dh (dhu) | 406 | 798 | 276 | 1.97 | 0.68 | 0.90 |
| bwd: dq/dk/dg/dbeta/dA_kk (wy + intra) | 1115 | 2585 | 1727 | 2.32 | 1.55 | 2.05 |
| bwd tail: cumsum, gate bwd, l2norm bwd | 372 | 380 | 266 | 1.02 | 0.71 | 0.95 |
| other (copies, fills, reductions) | 18 | 173 | 122 | 9.85 | 6.93 | 9.17 |
| **total** | **3862** | **6304** | **3818** | **1.63** | **0.99** | **1.31** |

### 8K x 96 heads (device us; ratio = other / ours, >1 = ours faster)

| stage | ours (BW) | FLA (BW) | FLA (A800) | FLA@BW / ours | FLA@A800 / ours | FLA@A800 x hw / ours |
|---|---:|---:|---:|---:|---:|---:|
| forward: prep (l2norm, gate, intra A, w/u) | 4767 | 6221 | 3676 | 1.30 | 0.77 | 1.02 |
| forward: recurrence h + output o | 1776 | 4579 | 1792 | 2.58 | 1.01 | 1.33 |
| bwd recompute: q/k norm, gate, w/u, h | 5249 | 5198 | 2550 | 0.99 | 0.49 | 0.64 |
| bwd: dA_qk + dv (dAv) | 1301 | 1062 | 590 | 0.82 | 0.45 | 0.60 |
| bwd: state gradient dh (dhu) | 1979 | 3916 | 1194 | 1.98 | 0.60 | 0.80 |
| bwd: dq/dk/dg/dbeta/dA_kk (wy + intra) | 12115 | 24167 | 13714 | 1.99 | 1.13 | 1.50 |
| bwd tail: cumsum, gate bwd, l2norm bwd | 2796 | 2905 | 2019 | 1.04 | 0.72 | 0.96 |
| other (copies, fills, reductions) | 55 | 1004 | 698 | 18.19 | 12.65 | 16.73 |
| **total** | **30038** | **49051** | **26234** | **1.63** | **0.87** | **1.16** |
