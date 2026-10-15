#!/bin/bash
# P-65 on bw7 (base g2_r5 = P-63d production): build, full gate (+FI=3), TP8 A/B on HCU 4 / 6
K=/public/home/tanbo/kra/examples/kda; T=/opt/kda_env/g2_p65_b7
cd /opt/kda_env && rm -rf g2_p65_b7 && cp -a g2_r5 g2_p65_b7 && cd g2_p65_b7 && rm -rf build
python3 $K/p65_lds_batch.py . || exit 1
python3 build_g2.py > /tmp/b65.log 2>&1 && echo build_ok || { grep -m5 error /tmp/b65.log; exit 1; }
L=/public/home/tanbo/codex_validation/locks/bw7; G=$K/bitwise_gate.py
r=$( exec 9>$L/hcu_4.lock; flock -w 900 9 || exit 75; export KDA_G2=1 HIP_VISIBLE_DEVICES=4 CACHE=/tmp/gate_b65
     bash $K/gate_all.sh $T "8192:12 8192:48" 2>&1
     KDA_G2_FI=3 PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover python3 $G dump --tree $T --T 8192 --H 12 --out /tmp/gate_b65/cand_fi3.pt > /tmp/gate_b65/fi3.log 2>&1
     echo "fi3 8192_12: $(python3 $G cmp /tmp/gate_b65/base_8192_12.pt /tmp/gate_b65/cand_fi3.pt | tail -1)" )
echo "$r" | grep -E "PASS|FAIL" | tr '\n' ' '; echo
[[ "$r" == *"GATE_ALL PASS"* ]] || { echo "GATE FAILED -> no A/B"; exit 1; }
X=/public/home/tanbo/xplat
bash $X/abk7.sh /opt/kda_env/g2_r5 $T 16384 12 4 $X/b7_p65_16384_12.txt &
bash $X/abk7.sh /opt/kda_env/g2_r5 $T 65536 12 7 $X/b7_p65_65536_12.txt &
wait; echo AB_DONE
