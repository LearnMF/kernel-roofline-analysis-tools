#!/bin/bash
# wall-clock A/B (CUDA events: fwd_ms, fb_ms = fwd+bwd, median of 7 per process), processes alternate
# A = base tree, B = candidate tree with extra env; N rounds.   bash wall_ab.sh GPU N "T:H ..." BASE CAND "ENV=.."
GPU=$1; N=$2; SHAPES=$3; A=$4; B=$5; BENV=$6
for s in $SHAPES; do
  T=${s%%:*}; H=${s##*:}
  for r in $(seq 1 $N); do
    for side in A B; do
      if [ $side = A ]; then tree=$A; envs=""; else tree=$B; envs="$BENV"; fi
      out=$(cd $tree && env $envs HIP_VISIBLE_DEVICES=$GPU KDA_G2=1 KRA_JITTER_SEED=$r PYTHONPATH=/public/home/tanbo/xplat/bwenv:$tree:/public/home/tanbo/codex_validation/stage_g2_bw7/takeover python3 tests/megatron/kda_op_bench_tp.py $T $H g2 7 2>/dev/null | grep "\"tag\": \"kda_op_bench\"")
      echo "{\"side\": \"$side\", \"T\": $T, \"H\": $H, \"r\": $r, \"data\": $out}"
    done
  done
done
