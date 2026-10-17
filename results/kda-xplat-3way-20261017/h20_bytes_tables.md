### 16K x 12: bytes (MB) and achieved bandwidth (GB/s) per stage

| stage | FLA bytes (A800 ncu) | ours bytes (BW pmc) | ours / FLA bytes | FLA on H20 us | FLA H20 GB/s | ours us | ours GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| forward prep (l2norm, gate, intra A, w/u) | 1195 | 848 | 0.71 | 642 | 1860 | 1014 | 836 |
| forward recurrence h + o | 695 | 380 | 0.55 | 582 | 1195 | 573 | 662 |
| bwd recompute (gate, w/u [+v_new], h) | 933 | 796 | 0.85 | 724 | 1289 | 833 | 956 |
| dA_qk + dv (dAv) | 223 | 281 | 1.26 | 86 | 2585 | 280 | 1005 |
| state gradient dh (dhu) | 409 | 405 | 0.99 | 438 | 934 | 660 | 613 |
| dq/dk/dg/dbeta/dA_kk (wy + intra) | 2845 | 1412 | 0.50 | 2324 | 1224 | 1785 | 791 |
| tail (cumsum, gate bwd, l2norm bwd) | 842 | 627 | 0.74 | 256 | 3293 | 636 | 986 |
| other | 280 | -- | -- | 108 | -- | -- | -- |
| **total** | **7423** | **4748** | **0.64** | **5160** | **1439** | **5781** | **821** |

### 16K x 48: bytes (MB) and achieved bandwidth (GB/s) per stage

| stage | FLA bytes (A800 ncu) | ours bytes (BW pmc) | ours / FLA bytes | FLA on H20 us | FLA H20 GB/s | ours us | ours GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| forward prep (l2norm, gate, intra A, w/u) | 4806 | 3428 | 0.71 | 2511 | 1914 | 3728 | 920 |
| forward recurrence h + o | 2735 | 2226 | 0.81 | 1486 | 1841 | 2397 | 928 |
| bwd recompute (gate, w/u [+v_new], h) | 3767 | 3188 | 0.85 | 2043 | 1844 | 3370 | 946 |
| dA_qk + dv (dAv) | 906 | 1116 | 1.23 | 330 | 2746 | 1090 | 1023 |
| state gradient dh (dhu) | 1623 | 1617 | 1.00 | 1304 | 1245 | 2017 | 802 |
| dq/dk/dg/dbeta/dA_kk (wy + intra) | 11496 | 5767 | 0.50 | 9328 | 1232 | 7474 | 772 |
| tail (cumsum, gate bwd, l2norm bwd) | 3409 | 2535 | 0.74 | 999 | 3412 | 2501 | 1014 |
| other | 1138 | -- | -- | 326 | -- | -- | -- |
| **total** | **29881** | **19876** | **0.67** | **18326** | **1631** | **22578** | **880** |

