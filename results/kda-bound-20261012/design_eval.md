
### 16K / H12 (TP8) -- production kernels sum 6.78 ms

| kernel | measured us | floor us (good engineering) | measured / floor | binding |
|---|---:|---:|---:|---|
| fwd_prep_a | 614 | 458 | 1.34 | valu |
| fwd_prep_b | 354 | 328 | 1.08 | TA |
| fwd_h_o | 717 | 362 | 1.98 | chain |
| l2n_apply | 156 | 184 | 0.85 | mem |
| head_gate | 175 | 181 | 0.97 | valu |
| wu | 646 | 580 | 1.11 | mem |
| fwd_h_bn | 537 | 294 | 1.83 | chain |
| dav | 278 | 253 | 1.10 | mem |
| dhu | 673 | 382 | 1.76 | TA |
| s4a | 595 | 421 | 1.41 | mem |
| s4b | 1325 | 1004 | 1.32 | TA |
| bwd_tail | 709 | 567 | 1.25 | mem |

| design | predicted us | vs production | H20 FLA / ours | A800 FLA / ours |
|---|---:|---:|---:|---:|
| production (measured) | 6779 | 1.00x | 0.76 | 1.10 |
| R  pipelined recurrences | 5874 | 1.15x | 0.88 | 1.27 |
| B4 fused gradient kernel | 4532 | 1.50x | 1.14 | 1.64 |
| R + B4 | 3627 | 1.87x | 1.42 | 2.05 |

(R: recurrences 1927 -> 1022 us;  B4: 2959 us of kernels -> 735 us, 4130 B/tok*head, VALU 19690 lane-ops/tok*head)

### 16K / H48 (TP2) -- production kernels sum 26.45 ms

| kernel | measured us | floor us (good engineering) | measured / floor | binding |
|---|---:|---:|---:|---|
| fwd_prep_a | 2312 | 1832 | 1.26 | valu |
| fwd_prep_b | 1471 | 1452 | 1.01 | TA |
| fwd_h_o | 1766 | 1392 | 1.27 | TA |
| l2n_apply | 629 | 734 | 0.86 | mem |
| head_gate | 678 | 722 | 0.94 | valu |
| wu | 2681 | 2320 | 1.16 | mem |
| fwd_h_bn | 1628 | 1300 | 1.25 | TA |
| dav | 1098 | 1012 | 1.08 | mem |
| dhu | 2524 | 1624 | 1.55 | TA |
| s4a | 2969 | 1924 | 1.54 | TA |
| s4b | 5943 | 4698 | 1.26 | TA |
| bwd_tail | 2750 | 2267 | 1.21 | mem |

| design | predicted us | vs production | H20 FLA / ours | A800 FLA / ours |
|---|---:|---:|---:|---:|
| production (measured) | 26450 | 1.00x | 0.69 | 0.99 |
| R  pipelined recurrences | 24101 | 1.10x | 0.76 | 1.09 |
| B4 fused gradient kernel | 16333 | 1.62x | 1.12 | 1.61 |
| R + B4 | 13984 | 1.89x | 1.31 | 1.88 |

(R: recurrences 5919 -> 3570 us;  B4: 12968 us of kernels -> 2939 us, 4130 B/tok*head, VALU 19690 lane-ops/tok*head)
