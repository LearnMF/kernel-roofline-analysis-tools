### 16384_12: pattern-aware baseline vs measured (us): gate_kernel 247/175, l2n_apply 144/156, g2_wu_kernel 657/639, g2_dav_kernel 241/277, wy_s4a 492/560, wy_s4b 1767/1283

| candidate | gate_kernel | l2n_apply | g2_wu_kernel | g2_dav_kernel | wy_s4a | wy_s4b | net us |
|---|---:|---:|---:|---:|---:|---:|---:|
| g native | +248.6 | +0.0 | +67.3 | +0.0 | +0.0 | -68.7 | **+247.2** |
| kn/qn native | +0.0 | +255.9 | +285.8 | +0.0 | +0.0 | -162.3 | **+379.3** |
| dA native (row reads) | +0.0 | +0.0 | +0.0 | +0.0 | -38.6 | -32.6 | **-71.1** |
| dAqk native (row reads) | +0.0 | +0.0 | +0.0 | -16.5 | +0.0 | -14.0 | **-30.5** |

### 16384_48: pattern-aware baseline vs measured (us): gate_kernel 966/672, l2n_apply 573/624, g2_wu_kernel 2542/2532, g2_dav_kernel 955/1041, wy_s4a 1915/2797, wy_s4b 6896/5898

| candidate | gate_kernel | l2n_apply | g2_wu_kernel | g2_dav_kernel | wy_s4a | wy_s4b | net us |
|---|---:|---:|---:|---:|---:|---:|---:|
| g native | +977.2 | +0.0 | +269.3 | +0.0 | +0.0 | -268.8 | **+977.7** |
| kn/qn native | +0.0 | +1021.1 | +1132.2 | +0.0 | +0.0 | -646.7 | **+1506.7** |
| dA native (row reads) | +0.0 | +0.0 | +0.0 | +0.0 | -153.0 | -128.8 | **-281.8** |
| dAqk native (row reads) | +0.0 | +0.0 | +0.0 | -65.5 | +0.0 | -55.2 | **-120.7** |
