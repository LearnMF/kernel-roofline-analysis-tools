#!/bin/bash
# same-tree env A/B of per-kernel device time + op total (kda_iter --kernels), processes alternate.
# bash ab_env.sh TREE GPU T:H "ENV_A" "ENV_B" N
TREE=$1; GPU=$2; S=$3; EA=$4; EB=$5; N=${6:-4}
T=${S%%:*}; H=${S##*:}
for r in $(seq 1 $N); do
  for side in A B; do
    if [ $side = A ]; then E="$EA"; else E="$EB"; fi
    out=$(cd /tmp && env $E HIP_VISIBLE_DEVICES=$GPU KDA_G2=1 KRA_JITTER_SEED=$r PYTHONPATH=/public/home/tanbo/xplat/bwenv:$TREE:/public/home/tanbo/codex_validation/stage_g2_bw7/takeover python3 $TREE/tests/megatron/kda_op_bench_tp.py $T $H g2 7 2>/dev/null | grep "\"tag\": \"kda_op_bench\"")
    echo "{\"side\": \"$side\", \"T\": $T, \"H\": $H, \"r\": $r, \"data\": $out}"
  done
done
