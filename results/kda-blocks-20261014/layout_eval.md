### 16384_12: pattern-aware baseline vs measured (us): gate_kernel 247/175, l2n_apply 144/156, g2_wu_kernel 657/639, g2_dav_kernel 241/277, wy_s4a 492/560, wy_s4b 1600/1283

| candidate | gate_kernel | l2n_apply | g2_wu_kernel | g2_dav_kernel | wy_s4a | wy_s4b | net us |
|---|---:|---:|---:|---:|---:|---:|---:|
| g native | +248.6 | +0.0 | +67.3 | +0.0 | +0.0 | -87.9 | **+228.0** |
| kn/qn native | +0.0 | +255.9 | +285.8 | +0.0 | +0.0 | -92.9 | **+448.7** |
| dA native (row reads) | +0.0 | +0.0 | +0.0 | +0.0 | -38.6 | -38.8 | **-77.3** |
| dA native (rows + transposed reads) | +0.0 | +0.0 | +0.0 | +0.0 | -38.6 | -42.6 | **-81.2** |
| dAqk native (row reads) | +0.0 | +0.0 | +0.0 | -16.5 | +0.0 | -16.6 | **-33.2** |
| dAqk native (rows + transposed reads) | +0.0 | +0.0 | +0.0 | -16.5 | +0.0 | -21.3 | **-37.9** |
| dA + dAqk native (all) | +0.0 | +0.0 | +0.0 | -16.5 | -38.6 | -63.8 | **-118.9** |

### 16384_48: pattern-aware baseline vs measured (us): gate_kernel 966/672, l2n_apply 573/624, g2_wu_kernel 2542/2532, g2_dav_kernel 955/1041, wy_s4a 1915/2797, wy_s4b 6231/5898

| candidate | gate_kernel | l2n_apply | g2_wu_kernel | g2_dav_kernel | wy_s4a | wy_s4b | net us |
|---|---:|---:|---:|---:|---:|---:|---:|
| g native | +977.2 | +0.0 | +269.3 | +0.0 | +0.0 | -347.7 | **+898.7** |
| kn/qn native | +0.0 | +1021.1 | +1132.2 | +0.0 | +0.0 | -372.5 | **+1780.8** |
| dA native (row reads) | +0.0 | +0.0 | +0.0 | +0.0 | -153.0 | -153.1 | **-306.1** |
| dA native (rows + transposed reads) | +0.0 | +0.0 | +0.0 | +0.0 | -153.0 | -168.2 | **-321.2** |
| dAqk native (row reads) | +0.0 | +0.0 | +0.0 | -65.5 | +0.0 | -65.6 | **-131.2** |
| dAqk native (rows + transposed reads) | +0.0 | +0.0 | +0.0 | -65.5 | +0.0 | -84.2 | **-149.7** |
| dA + dAqk native (all) | +0.0 | +0.0 | +0.0 | -65.5 | -153.0 | -251.9 | **-470.5** |
