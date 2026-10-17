#!/bin/bash
K=/public/home/tanbo/kra/examples/kda; T=/opt/kda_env/g2_p88; L=/public/home/tanbo/codex_validation/locks/bw7; G=${1:-1}
cd /opt/kda_env && rm -rf g2_p88 && cp -a g2_r5 g2_p88 && cd g2_p88 && rm -rf build
python3 $K/p88_s4_hf.py csrc/g2 || exit 1
python3 build_g2.py > /tmp/b88.log 2>&1 && echo build_ok || { grep -m8 error /tmp/b83.log; exit 1; }
GT=$K/bitwise_gate.py
r=$( exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75; export KDA_G2=1 HIP_VISIBLE_DEVICES=$G CACHE=/tmp/gate_b88 KDA_G2_S4_HF=1
     bash $K/gate_all.sh $T "8192:12 8192:48 8192:96" 2>&1
     KDA_G2_FI=3 PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover python3 $GT dump --tree $T --T 8192 --H 12 --out /tmp/gate_b88/cand_fi3.pt > /tmp/gate_b88/fi3.log 2>&1
     echo "fi3 8192_12: $(python3 $GT cmp /tmp/gate_b88/base_8192_12.pt /tmp/gate_b88/cand_fi3.pt | tail -1)" )
echo "$r" | grep -E "PASS|FAIL" | tr '\n' ' '; echo
[[ "$r" == *"GATE_ALL PASS"* ]] || { echo "GATE FAILED -> no A/B"; echo "$r" | tail -30; exit 1; }
( exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75; 
  export KDA_G2=1 HIP_VISIBLE_DEVICES=$G PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover; cd /tmp
  for TH in "16384 96" "8192 96" "16384 48" "16384 12" "32768 96"; do set -- $TH
    python3 $K/op_env_time.py --tree $T --fn wy_s4_split --T $1 --H $2 --env KDA_G2_S4_HF --vals 0,1 --reps 12 2>&1 | grep -E " us |rror"
    python3 $K/pipe_time.py --tree $T --T $1 --H $2 --env KDA_G2_S4_HF --reps 10 2>&1 | grep "T="; done )
echo P88_DONE
