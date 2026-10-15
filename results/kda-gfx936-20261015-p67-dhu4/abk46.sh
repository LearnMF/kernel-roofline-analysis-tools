#!/bin/bash
# bw46 (tanbo_kda_b46) 5-round per-kernel A/B under the shared HCU lock: abk7.sh A B T H GPU OUT
A=$1; B=$2; T=$3; H=$4; G=$5; OUT=$6
L=/public/home/tanbo/codex_validation/locks
exec 8>"$L/bw46/hcu_${G}.lock"; flock -w 3600 8 || { echo LOCK_BUSY > $OUT; exit 75; }
export KDA_G2=1 KDA_SKIP_FLA_PRUNE=1 HIP_VISIBLE_DEVICES=$G JITTER=1
cd /tmp && bash /public/home/tanbo/kra/examples/kda/ab_kernels.sh $A $B $T $H 5 6 /tmp/abk46_$G > $OUT 2>&1; echo DONE >> $OUT
