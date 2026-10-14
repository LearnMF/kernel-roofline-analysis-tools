#!/bin/bash
# P-61e on bw7: build from g2_r5 (= bw52 g2_opt 9802dae), gate vs g2_r5 (+ FI=3 path), then 4-shape A/B on HCU 4-7
cd /opt/kda_env && rm -rf g2_p61e_b7 && cp -a g2_r5 g2_p61e_b7 && cd g2_p61e_b7 && rm -rf build
python3 /public/home/tanbo/kra/examples/kda/p61e_kn_perm.py . || exit 1
python3 build_g2.py > /tmp/b61e.log 2>&1 && echo build_ok || { grep -m8 -E "error|Error" /tmp/b61e.log; exit 1; }
L=/public/home/tanbo/codex_validation/locks/bw7; G=/public/home/tanbo/kra/examples/kda/bitwise_gate.py
r=$( exec 9>$L/hcu_4.lock; flock -w 900 9 || exit 75; export KDA_G2=1 HIP_VISIBLE_DEVICES=4 CACHE=/tmp/gate_b61e
     bash /public/home/tanbo/kra/examples/kda/gate_all.sh /opt/kda_env/g2_p61e_b7 "8192:12 8192:48" 2>&1
     KDA_G2_FI=3 PYTHONPATH=/opt/kda_env:/opt/kda_env/g2_p61e_b7:/opt/kda_env/takeover python3 $G dump --tree /opt/kda_env/g2_p61e_b7 \
        --T 8192 --H 12 --out /tmp/gate_b61e/cand_fi3.pt > /tmp/gate_b61e/fi3.log 2>&1 || echo "fi3: candidate run FAILED"
     echo "fi3 8192_12: $(python3 $G cmp /tmp/gate_b61e/base_8192_12.pt /tmp/gate_b61e/cand_fi3.pt | tail -1)" )
echo "$r" | grep -E "PASS|FAIL" | tr '\n' ' '; echo
[[ "$r" == *"GATE_ALL PASS"* && "$r" == *"fi3 8192_12: GATE PASS"* ]] || { echo "GATE FAILED -> no A/B"; exit 1; }
X=/public/home/tanbo/xplat; g=4
for spec in 16384:12 65536:12 16384:48 8192:96; do T=${spec%:*}; H=${spec#*:}
  bash $X/abk7.sh /opt/kda_env/g2_r5 /opt/kda_env/g2_p61e_b7 $T $H $g $X/b7_p61e_${T}_${H}.txt & g=$((g+1)); done
wait; echo AB_ALL_DONE
