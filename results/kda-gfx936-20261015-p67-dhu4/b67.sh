#!/bin/bash
# P-67 on bw46 (tanbo_kda_b46): build from g2_r5 (= production P-65), full gate (+FI=3), TP8 A/B (HCU 1/2), dhu SQTT (HCU3)
K=/public/home/tanbo/kra/examples/kda; T=/opt/kda_env/g2_p67; X=/public/home/tanbo/xplat; L=/public/home/tanbo/codex_validation/locks/bw46
cd /opt/kda_env && rm -rf g2_p67 && cp -a g2_r5 g2_p67 && cd g2_p67 && rm -rf build
python3 $K/p67_dhu4_tp8.py . || exit 1
python3 build_g2.py > /tmp/b67.log 2>&1 && echo build_ok || { grep -m5 error /tmp/b67.log; exit 1; }
G=$K/bitwise_gate.py
r=$( exec 9>$L/hcu_0.lock; flock -w 900 9 || exit 75; export KDA_G2=1 HIP_VISIBLE_DEVICES=0 CACHE=/tmp/gate_b67
     bash $K/gate_all.sh $T "8192:12 8192:48" 2>&1
     KDA_G2_FI=3 PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover python3 $G dump --tree $T --T 8192 --H 12 --out /tmp/gate_b67/cand_fi3.pt > /tmp/gate_b67/fi3.log 2>&1
     echo "fi3 8192_12: $(python3 $G cmp /tmp/gate_b67/base_8192_12.pt /tmp/gate_b67/cand_fi3.pt | tail -1)" )
echo "$r" | grep -E "PASS|FAIL" | tr '\n' ' '; echo
[[ "$r" == *"GATE_ALL PASS"* ]] || { echo "GATE FAILED -> no A/B"; exit 1; }
bash $X/abk46.sh /opt/kda_env/g2_r5 $T 16384 12 1 $X/b46_p67_16384_12.txt &
bash $X/abk46.sh /opt/kda_env/g2_r5 $T 65536 12 2 $X/b46_p67_65536_12.txt &
wait; echo AB_DONE
