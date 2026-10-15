#!/bin/bash
# P-72 on bw46: bwd_tail phase B 2 rows / instruction
K=/public/home/tanbo/kra/examples/kda; T=/opt/kda_env/g2_p72; X=/public/home/tanbo/xplat; L=/public/home/tanbo/codex_validation/locks/bw46
cd /opt/kda_env && rm -rf g2_p72 && cp -a g2_r5 g2_p72 && cd g2_p72 && rm -rf build
python3 $K/p72_tail_wide.py . || exit 1
python3 build_g2.py > /tmp/b72.log 2>&1 && echo build_ok || { grep -m8 -E " error" /tmp/b72.log; exit 1; }
G=$K/bitwise_gate.py
r=$( exec 9>$L/hcu_0.lock; flock -w 900 9 || exit 75; export KDA_G2=1 HIP_VISIBLE_DEVICES=0 CACHE=/tmp/gate_b72
     bash $K/gate_all.sh $T "8192:12 8192:48" 2>&1
     KDA_G2_FI=3 PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover python3 $G dump --tree $T --T 8192 --H 12 --out /tmp/gate_b72/cand_fi3.pt > /tmp/gate_b72/fi3.log 2>&1
     echo "fi3 8192_12: $(python3 $G cmp /tmp/gate_b72/base_8192_12.pt /tmp/gate_b72/cand_fi3.pt | tail -1)" )
echo "$r" | grep -E "PASS|FAIL" | tr '\n' ' '; echo
[[ "$r" == *"GATE_ALL PASS"* ]] || { echo "GATE FAILED -> no A/B"; exit 1; }
g=1; for TH in "16384 12" "65536 12" "16384 48" "8192 96"; do set -- $TH
  bash $X/abk46.sh /opt/kda_env/g2_r5 $T $1 $2 $g $X/b46_p72_$1_$2.txt & g=$((g+1)); [ $g = 4 ] && g=7; done
wait; echo AB_DONE
