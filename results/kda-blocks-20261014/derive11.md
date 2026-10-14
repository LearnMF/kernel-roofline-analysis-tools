
### 16384_12

| kernel | kind | predicted us | measured us | error | hbm_frac |
|---|---|---:|---:|---:|---:|
| fwd_prep_a#1 | chunk (trace, res 3) | 644 | 604 | +7% | 0.94 |
| fwd_prep_b#1 | chunk (trace, res 3) | 289 | 349 | -17% | 1.00 |
| l2n_apply#1 | chunk (trace, res 10) | 169 | 156 | +9% | 0.68 |
| gate#1 | chunk (trace, res 8) | 241 | 175 | +37% | 0.89 |
| wu#1 | chunk (trace, res 2) | 659 | 639 | +3% | 1.00 |
| dav#1 | chunk (trace, res 3) | 253 | 277 | -9% | 0.77 |
| wy_s4a#1 | chunk (trace, res 2) | 472 | 560 | -16% | 1.00 |
| wy_s4b#1 | chunk (trace, res 2) | 1008 | 1283 | -21% | 0.82 |
| bwd_tail#1 | chunk (trace, res 6) | 599 | 643 | -7% | 1.00 |
| fwd_h_v8o4#1 | recurrence (step) | 537 | 658 | -18% | 0.27 |
| fwd_h_v8o4#2 | recurrence (step) | 515 | 503 | +2% | 0.28 |
| dhu_v8b#1 | recurrence (step) | 545 | 656 | -17% | 0.25 |
| **sum of the kernels above** | | 5930 | 6504 | -9% | |

### 16384_48

| kernel | kind | predicted us | measured us | error | hbm_frac |
|---|---|---:|---:|---:|---:|
| fwd_prep_a#1 | chunk (trace, res 3) | 2549 | 2288 | +11% | 0.96 |
| fwd_prep_b#1 | chunk (trace, res 3) | 1143 | 1439 | -21% | 1.00 |
| l2n_apply#1 | chunk (trace, res 10) | 672 | 624 | +8% | 0.68 |
| gate#1 | chunk (trace, res 8) | 939 | 672 | +40% | 0.89 |
| wu#1 | chunk (trace, res 2) | 2564 | 2532 | +1% | 1.00 |
| dav#1 | chunk (trace, res 3) | 997 | 1041 | -4% | 0.77 |
| wy_s4a#1 | chunk (trace, res 2) | 1839 | 2797 | -34% | 1.00 |
| wy_s4b#1 | chunk (trace, res 2) | 3903 | 5898 | -34% | 0.82 |
| bwd_tail#1 | chunk (trace, res 6) | 2204 | 2584 | -15% | 1.00 |
| dhu_v8b4#1 | recurrence (step) | 1931 | 2328 | -17% | 0.31 |
| **sum of the kernels above** | | 18741 | 22203 | -16% | |

22 kernel-shape points: 16 within +-20%, 18 within +-30%
