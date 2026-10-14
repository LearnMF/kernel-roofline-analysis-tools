
### 16384_12

| kernel | kind | predicted us | measured us | error | hbm_frac |
|---|---|---:|---:|---:|---:|
| fwd_prep_a#1 | chunk (trace) | 644 | 604 | +7% | 0.94 |
| fwd_prep_b#1 | chunk (trace) | 289 | 349 | -17% | 1.00 |
| l2n_apply#1 | chunk (trace) | 181 | 156 | +16% | 0.68 |
| gate#1 | chunk (trace) | 245 | 175 | +40% | 0.89 |
| wu#1 | chunk (trace) | 659 | 639 | +3% | 1.00 |
| dav#1 | chunk (trace) | 253 | 277 | -9% | 0.77 |
| wy_s4a#1 | chunk (trace) | 472 | 560 | -16% | 1.00 |
| wy_s4b#1 | chunk (trace) | 1008 | 1283 | -21% | 0.82 |
| bwd_tail#1 | chunk (trace) | 626 | 643 | -3% | 1.00 |
| fwd_h_v8o4#1 | recurrence (step) | 537 | 658 | -18% | 0.27 |
| fwd_h_v8o4#2 | recurrence (step) | 515 | 503 | +2% | 0.28 |
| dhu_v8b#1 | recurrence (step) | 545 | 656 | -17% | 0.25 |
| **sum of the kernels above** | | 5973 | 6504 | -8% | |

### 16384_48

| kernel | kind | predicted us | measured us | error | hbm_frac |
|---|---|---:|---:|---:|---:|
| fwd_prep_a#1 | chunk (trace) | 2549 | 2288 | +11% | 0.96 |
| fwd_prep_b#1 | chunk (trace) | 1143 | 1439 | -21% | 1.00 |
| l2n_apply#1 | chunk (trace) | 720 | 624 | +15% | 0.68 |
| gate#1 | chunk (trace) | 965 | 672 | +43% | 0.89 |
| wu#1 | chunk (trace) | 2564 | 2532 | +1% | 1.00 |
| dav#1 | chunk (trace) | 997 | 1041 | -4% | 0.77 |
| wy_s4a#1 | chunk (trace) | 1839 | 2797 | -34% | 1.00 |
| wy_s4b#1 | chunk (trace) | 3903 | 5898 | -34% | 0.82 |
| bwd_tail#1 | chunk (trace) | 2474 | 2584 | -4% | 1.00 |
| dhu_v8b4#1 | recurrence (step) | 1931 | 2328 | -17% | 0.31 |
| **sum of the kernels above** | | 19086 | 22203 | -14% | |

22 kernel-shape points: 16 within +-20%, 18 within +-30%
