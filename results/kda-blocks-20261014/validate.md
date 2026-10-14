## 1. recurrence step (ISA simulation vs measured)

| instance | hbm_frac | predicted us/step | measured us/step | error |
|---|---:|---:|---:|---:|
| fwd MODE 1 (fwd_h_o) | 0.267 | 2.087 | 2.571 | -19% |
| recompute MODE 4 (fwd_h_bn) | 0.276 | 2.009 | 1.966 | +2% |
| dhu 8-wave | 0.252 | 2.132 | 2.564 | -17% |
| dhu 4-wave (P-24 throttled, 1 CTA/CU) | 0.314 | 2.517 | 3.031 | -17% |

## 2. bwd_tail TA cycles per wave (inventory x library costs, data from HBM) vs the card

| shape | predicted | measured (TA busy / wave) | error | inventory VMEM / wave vs PMC |
|---|---:|---:|---:|---|
| 16384_12 | 6209 | 5907 | +5% | 148 vs 194.8 |
| 16384_48 | 5768 | 6155 | -6% | 148 vs 194.8 |

## 3. sanity vs machine ceilings (machines/gfx936-bw1000.json)

| quantity | library | ceiling | ratio |
|---|---:|---:|---:|
| HBM read (1 KB rows) | 1.235 TB/s | 1.3392 TB/s | 0.92 |
| HBM write (1 KB rows) | 1.109 TB/s | 1.1005 TB/s | 1.01 |
| L2 read (1 KB rows) | 5.568 TB/s | 6.1 TB/s | 0.91 |
| LDS (best op) | 13.524 TB/s | 15.2 TB/s | 0.89 |
| bf16 MMAC | 451.972 TFLOPS | 475.96 TFLOPS | 0.95 |
