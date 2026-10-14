#!/bin/bash
# P-61b A/B on bw7 (bw52 is shared with an 8-GPU job): build from g2_r5 (= bw52 g2_opt 9802dae), 4 shapes on HCU 4-7
set -e
cd /opt/kda_env && rm -rf g2_p61b_b7 && cp -a g2_r5 g2_p61b_b7 && cd g2_p61b_b7 && rm -rf build
python3 /public/home/tanbo/kra/examples/kda/p61_kn_perm.py . && python3 /public/home/tanbo/kra/examples/kda/p61b_kn_perm_bwd.py .
python3 build_g2.py > /tmp/b61b.log 2>&1 && echo build_ok
for f in csrc/g2/g2_fwd_v8.cuh csrc/g2/g2_fwd_prep.cuh csrc/g2/g2_wu.cuh hip_kda/ops/k3_g2.py; do cmp $f /public/home/tanbo/g2_p61b/$f && echo "same $f"; done
set +e
X=/public/home/tanbo/xplat; g=4
for spec in 16384:12 65536:12 16384:48 8192:96; do T=${spec%:*}; H=${spec#*:}
  bash $X/abk7.sh /opt/kda_env/g2_r5 /opt/kda_env/g2_p61b_b7 $T $H $g $X/b7_p61b_${T}_${H}.txt & g=$((g+1)); done
wait; echo AB_ALL_DONE
