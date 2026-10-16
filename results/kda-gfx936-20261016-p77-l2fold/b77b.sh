#!/bin/bash
# P-77 on bw46: build, full bitwise gates (fold on, vs production g2_r5), FI=3, then in-process wall A/B
# (KDA_G2_SAVE_QKN 0 vs 1 in the same tree) + per-kernel A/B vs production
K=/public/home/tanbo/kra/examples/kda; T=/opt/kda_env/g2_p77b; X=/public/home/tanbo/xplat; L=/public/home/tanbo/codex_validation/locks/bw46
G=${1:-7}
cd /opt/kda_env && rm -rf g2_p77b && cp -a g2_r5 g2_p77b && cd g2_p77b && rm -rf build
python3 $K/p77b_save_qkn.py . || exit 1
python3 build_g2.py > /tmp/b77b.log 2>&1 && echo build_ok || { grep -m8 -E " error|error:" /tmp/b77.log; exit 1; }
GT=$K/bitwise_gate.py
r=$( exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75; export KDA_G2=1 HIP_VISIBLE_DEVICES=$G CACHE=/tmp/gate_b77b
     bash $K/gate_all.sh $T "8192:12 8192:48" 2>&1
     KDA_G2_FI=3 PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover python3 $GT dump --tree $T --T 8192 --H 12 --out /tmp/gate_b77b/cand_fi3.pt > /tmp/gate_b77b/fi3.log 2>&1
     echo "fi3 8192_12: $(python3 $GT cmp /tmp/gate_b77b/base_8192_12.pt /tmp/gate_b77b/cand_fi3.pt | tail -1)" )
echo "$r" | grep -E "PASS|FAIL" | tr '\n' ' '; echo
[[ "$r" == *"GATE_ALL PASS"* ]] || { echo "GATE FAILED -> no A/B"; echo "$r" | tail -30; exit 1; }
( exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75; export KDA_G2=1 HIP_VISIBLE_DEVICES=$G PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover; cd /tmp
  for TH in "16384 12" "65536 12" "16384 48" "8192 96"; do set -- $TH; python3 $K/pipe_time.py --tree $T --T $1 --H $2 --env KDA_G2_SAVE_QKN --reps 12 2>&1 | grep "T="; done )
g=1; for TH in "16384 12" "65536 12" "16384 48"; do set -- $TH
  bash $X/abk46.sh /opt/kda_env/g2_r5 $T $1 $2 $g $X/b46_p77b_$1_$2.txt & g=$((g+1)); done
wait; echo AB_DONE
