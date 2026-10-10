
### 8K x 12 heads (device us; ratio = other / ours, >1 = ours faster)

| stage | ours (BW) | FLA (BW) | FLA (A800) | FLA@BW / ours | FLA@A800 / ours | FLA@A800 x hw / ours |
|---|---:|---:|---:|---:|---:|---:|
| forward: prep (l2norm, gate, intra A, w/u) | 661 | 815 | 493 | 1.23 | 0.74 | 0.99 |
| forward: recurrence h + output o | 358 | 605 | 382 | 1.69 | 1.07 | 1.41 |
| bwd recompute: q/k norm, gate, w/u, h | 910 | 809 | 476 | 0.89 | 0.52 | 0.69 |
| bwd: dA_qk + dv (dAv) | 283 | 139 | 77 | 0.49 | 0.27 | 0.36 |
| bwd: state gradient dh (dhu) | 434 | 798 | 276 | 1.84 | 0.64 | 0.84 |
| bwd: dq/dk/dg/dbeta/dA_kk (wy + intra) | 1107 | 2585 | 1727 | 2.34 | 1.56 | 2.06 |
| bwd tail: cumsum, gate bwd, l2norm bwd | 371 | 380 | 266 | 1.02 | 0.72 | 0.95 |
| other (copies, fills, reductions) | 18 | 173 | 122 | 9.85 | 6.93 | 9.17 |
| **total** | **4142** | **6304** | **3818** | **1.52** | **0.92** | **1.22** |

### 8K x 96 heads (device us; ratio = other / ours, >1 = ours faster)

| stage | ours (BW) | FLA (BW) | FLA (A800) | FLA@BW / ours | FLA@A800 / ours | FLA@A800 x hw / ours |
|---|---:|---:|---:|---:|---:|---:|
| forward: prep (l2norm, gate, intra A, w/u) | 4763 | 6221 | 3676 | 1.31 | 0.77 | 1.02 |
| forward: recurrence h + output o | 2793 | 4579 | 1792 | 1.64 | 0.64 | 0.85 |
| bwd recompute: q/k norm, gate, w/u, h | 7497 | 5198 | 2550 | 0.69 | 0.34 | 0.45 |
| bwd: dA_qk + dv (dAv) | 2391 | 1062 | 590 | 0.44 | 0.25 | 0.33 |
| bwd: state gradient dh (dhu) | 2512 | 3916 | 1194 | 1.56 | 0.48 | 0.63 |
| bwd: dq/dk/dg/dbeta/dA_kk (wy + intra) | 12134 | 24167 | 13714 | 1.99 | 1.13 | 1.49 |
| bwd tail: cumsum, gate bwd, l2norm bwd | 2801 | 2905 | 2019 | 1.04 | 0.72 | 0.95 |
| other (copies, fills, reductions) | 54 | 1004 | 698 | 18.57 | 12.91 | 17.07 |
| **total** | **34946** | **49051** | **26234** | **1.40** | **0.75** | **0.99** |
