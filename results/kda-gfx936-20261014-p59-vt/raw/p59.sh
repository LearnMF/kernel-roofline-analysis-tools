#!/bin/bash
# kra P-59 probe on bw52 (container tanbo_mega_bw52): bitwise gate vs production g2_opt, then 4-shape kernel A/B.
X=/public/home/tanbo/xplat; L=/public/home/tanbo/codex_validation/locks
exec 9>$L/hcu_0.lock; flock -n 9 || { echo LOCK_BUSY_0; exit 75; }
export KDA_G2=1 HIP_VISIBLE_DEVICES=0 PP_PRE=$X/bwenv PP_POST=/public/home/tanbo/codex_validation/stage_g2_bw7/takeover BASE=/public/home/tanbo/g2_opt CACHE=/tmp/gate_p59
r=$(bash /public/home/tanbo/kra/examples/kda/gate_all.sh /public/home/tanbo/g2_p59 "8192:12 8192:48" 2>&1)
echo "$r" | grep -E "PASS|FAIL|differ|mismatch|max" | tr '\n' ' '; echo
exec 9>&-
[[ "$r" == *"GATE_ALL PASS"* ]] || { echo "GATE FAILED -> no A/B"; exit 1; }
g=1
for spec in 16384:12 65536:12 16384:48 8192:96; do T=${spec%:*}; H=${spec#*:}; rm -f $X/p59_${T}_${H}.txt
  (exec 8>$L/hcu_$g.lock; flock -n 8 || { echo LOCK_BUSY > $X/p59_${T}_${H}.txt; exit 75; }
   bash $X/abk5.sh /public/home/tanbo/g2_opt /public/home/tanbo/g2_p59 $T $H $g $X/p59_${T}_${H}.txt) &
  g=$((g+2)); done
wait; echo AB_ALL_DONE
