#!/bin/bash
# Per-round A800 tracker, BW side (inside the bw52 container): T in {8K,16K,32K} x TP in {1,2,4,8}
# (H = 96/48/24/12) for ONE tree (default: production).  Device time = hipprof kernel sum per
# fwd+bwd window, N processes per shape with placement jitter (R20), run in parallel over GPUs
# (device time is insensitive to host contention); wall time = CUDA/HIP events around one fwd+bwd
# (kda_op_bench_tp.py arm g2, median of 7), N processes, SEQUENTIAL on one GPU.
#   bash round_vs_a800.sh OUTDIR [TREE=/public/home/tanbo/g2_opt] [N=3] [GPUS="1 3 4 5 6 7"] [WALL_GPU=2]
OUT=$1; TREE=${2:-/public/home/tanbo/g2_opt}; N=${3:-3}; GPUS=${4:-"1 3 4 5 6 7"}; WALL_GPU=${5:-2}
SHAPES="8192:12 8192:24 8192:48 8192:96 16384:12 16384:24 16384:48 16384:96 32768:12 32768:24 32768:48 32768:96"
KRA=/public/home/tanbo/kra
mkdir -p "$OUT/dev"; rm -f "$OUT/wall.jsonl"
export PYTHONPATH=/public/home/tanbo/xplat/bwenv:$TREE:/public/home/tanbo/codex_validation/stage_g2_bw7/takeover
export KDA_G2=1 TRITON_CACHE_DIR=/tmp/triton_round PYTHONDONTWRITEBYTECODE=1
# device time: shapes dealt round-robin over GPUs, each GPU sequential
i=0; declare -A Q
for s in $SHAPES; do g=$(echo $GPUS | cut -d' ' -f$((i % $(echo $GPUS | wc -w) + 1))); Q[$g]="${Q[$g]} $s"; i=$((i + 1)); done
for g in "${!Q[@]}"; do
  (
    for s in ${Q[$g]}; do
      T=${s%%:*}; H=${s##*:}
      for r in $(seq 1 "$N"); do
        tag="g2_${T}_${H}_r${r}"; rm -f /tmp/rva_$tag.*
        KRA_JITTER_SEED=$r HIP_VISIBLE_DEVICES=$g hipprof --hip-trace --output-type 0 -o /tmp/rva_$tag \
          python3 $KRA/examples/kda/kda_iter.py --tree "$TREE" --arm g2 --T "$T" --H "$H" --iters 5 > /dev/null 2>&1
        cp /tmp/rva_$tag.json "$OUT/dev/xpd_${tag}.json" 2>/dev/null; rm -f /tmp/rva_$tag.*
      done
    done
  ) &
done
# wall time: sequential on one GPU (host contention would inflate it), concurrent with the above
(
  cd "$TREE"
  for s in $SHAPES; do
    T=${s%%:*}; H=${s##*:}
    for r in $(seq 1 "$N"); do
      HIP_VISIBLE_DEVICES=$WALL_GPU python3 tests/megatron/kda_op_bench_tp.py "$T" "$H" g2 7 2>/dev/null \
        | grep '"tag": "kda_op_bench"' | sed "s/^/{\"run\": $r, \"host\": \"bw52\", \"data\": /; s/\$/}/" >> "$OUT/wall.jsonl"
    done
  done
) &
wait
cd $KRA && PYTHONPATH=$KRA python3 examples/kda/xplat_bw_dev_parse.py "$OUT/dev" > "$OUT/dev.json"
echo ROUND_VS_A800_DONE
