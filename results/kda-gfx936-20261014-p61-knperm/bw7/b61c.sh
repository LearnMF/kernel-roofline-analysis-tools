#!/bin/bash
# P-61c on bw7: build from g2_r5 (= bw52 g2_opt 9802dae), bitwise gate vs g2_r5, then 4-shape A/B on HCU 4-7
cd /opt/kda_env && rm -rf g2_p61c_b7 && cp -a g2_r5 g2_p61c_b7 && cd g2_p61c_b7 && rm -rf build
python3 /public/home/tanbo/kra/examples/kda/p61c_kn_load_perm.py . || exit 1
python3 build_g2.py > /tmp/b61c.log 2>&1 && echo build_ok || { grep -m5 error /tmp/b61c.log; exit 1; }
L=/public/home/tanbo/codex_validation/locks/bw7
r=$( exec 9>$L/hcu_4.lock; flock -w 600 9 || exit 75; KDA_G2=1 HIP_VISIBLE_DEVICES=4 CACHE=/tmp/gate_b61c \
     bash /public/home/tanbo/kra/examples/kda/gate_all.sh /opt/kda_env/g2_p61c_b7 "8192:12 8192:48" 2>&1 )
echo "$r" | grep -E "PASS|FAIL" | tr '\n' ' '; echo
[[ "$r" == *"GATE_ALL PASS"* ]] || { echo "GATE FAILED -> no A/B"; exit 1; }
X=/public/home/tanbo/xplat; g=4
for spec in 16384:12 65536:12 16384:48 8192:96; do T=${spec%:*}; H=${spec#*:}
  bash $X/abk7.sh /opt/kda_env/g2_r5 /opt/kda_env/g2_p61c_b7 $T $H $g $X/b7_p61c_${T}_${H}.txt & g=$((g+1)); done
wait; echo AB_ALL_DONE
