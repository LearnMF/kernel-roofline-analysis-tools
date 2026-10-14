#!/bin/bash
# ab_only.sh CAND_TREE TAG : 4-shape kernel A/B vs production g2_opt, GPUs 0,2,4,6 with locks
X=/public/home/tanbo/xplat; L=/public/home/tanbo/codex_validation/locks; C=$1; TAG=$2; g=0
for spec in 16384:12 65536:12 16384:48 8192:96; do T=${spec%:*}; H=${spec#*:}; rm -f $X/${TAG}_${T}_${H}.txt
  (exec 8>$L/hcu_$g.lock; flock -n 8 || { echo LOCK_BUSY > $X/${TAG}_${T}_${H}.txt; exit 75; }
   bash $X/abk5.sh /public/home/tanbo/g2_opt $C $T $H $g $X/${TAG}_${T}_${H}.txt) &
  g=$((g+2)); done
wait; echo AB_ALL_DONE
