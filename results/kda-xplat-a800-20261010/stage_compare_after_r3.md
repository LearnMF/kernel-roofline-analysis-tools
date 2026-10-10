
### 8K x 12 heads (device us; ratio = other / ours, >1 = ours faster)

| stage | ours (BW) | FLA (BW) | FLA (A800) | FLA@BW / ours | FLA@A800 / ours | FLA@A800 x hw / ours |
|---|---:|---:|---:|---:|---:|---:|
| forward: prep (l2norm, gate, intra A, w/u) | 662 | 815 | 493 | 1.23 | 0.74 | 0.98 |
| forward: recurrence h + output o | 354 | 605 | 382 | 1.71 | 1.08 | 1.43 |
| bwd recompute: q/k norm, gate, w/u, h | 814 | 809 | 476 | 0.99 | 0.59 | 0.77 |
| bwd: dA_qk + dv (dAv) | 193 | 139 | 77 | 0.72 | 0.40 | 0.53 |
| bwd: state gradient dh (dhu) | 434 | 798 | 276 | 1.84 | 0.64 | 0.84 |
| bwd: dq/dk/dg/dbeta/dA_kk (wy + intra) | 1089 | 2585 | 1727 | 2.37 | 1.58 | 2.10 |
| bwd tail: cumsum, gate bwd, l2norm bwd | 371 | 380 | 266 | 1.02 | 0.72 | 0.95 |
| other (copies, fills, reductions) | 18 | 173 | 122 | 9.85 | 6.93 | 9.17 |
| **total** | **3935** | **6304** | **3818** | **1.60** | **0.97** | **1.28** |

### 8K x 96 heads (device us; ratio = other / ours, >1 = ours faster)

| stage | ours (BW) | FLA (BW) | FLA (A800) | FLA@BW / ours | FLA@A800 / ours | FLA@A800 x hw / ours |
|---|---:|---:|---:|---:|---:|---:|
| forward: prep (l2norm, gate, intra A, w/u) | 4657 | 6221 | 3676 | 1.34 | 0.79 | 1.04 |
| forward: recurrence h + output o | 2216 | 4579 | 1792 | 2.07 | 0.81 | 1.07 |
| bwd recompute: q/k norm, gate, w/u, h | 5707 | 5198 | 2550 | 0.91 | 0.45 | 0.59 |
| bwd: dA_qk + dv (dAv) | 1905 | 1062 | 590 | 0.56 | 0.31 | 0.41 |
| bwd: state gradient dh (dhu) | 2463 | 3916 | 1194 | 1.59 | 0.48 | 0.64 |
| bwd: dq/dk/dg/dbeta/dA_kk (wy + intra) | 10807 | 24167 | 13714 | 2.24 | 1.27 | 1.68 |
| bwd tail: cumsum, gate bwd, l2norm bwd | 2644 | 2905 | 2019 | 1.10 | 0.76 | 1.01 |
| other (copies, fills, reductions) | 54 | 1004 | 698 | 18.59 | 12.93 | 17.10 |
| **total** | **30453** | **49051** | **26234** | **1.61** | **0.86** | **1.14** |
