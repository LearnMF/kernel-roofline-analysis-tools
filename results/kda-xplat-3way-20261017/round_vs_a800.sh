#!/bin/bash
# Per-round A800 tracker, BW side (inside the bw52 container): T in {8K,16K,32K,64K} x TP in {1,2,4,8}
# (H = 96/48/24/12).  Device time = hipprof kernel sum per fwd+bwd window, N processes per shape
# with placement jitter (R20), run in parallel over GPUs (device time is insensitive to host
# contention).  With PREV set, the previous round's production is measured in the SAME session,
# interleaved process by process (prev, now, prev, now, ... with the same jitter seeds), so the
# round-over-round column is an A/B and not a cross-session comparison.  Wall time = CUDA/HIP events
# around one fwd+bwd (kda_op_bench_tp.py arm g2, median of 7), NW processes, SEQUENTIAL on one GPU.
#   PREV=<tree> bash round_vs_a800.sh OUTDIR [TREE=/public/home/tanbo/g2_opt] [N=5] [GPUS="1 3 4 5 6 7"] [WALL_GPU=2] [NW=3]
OUT=$1; TREE=${2:-/public/home/tanbo/g2_opt}; N=${3:-5}; GPUS=${4:-"1 3 4 5 6 7"}; WALL_GPU=${5:-2}; NW=${6:-3}
SHAPES=${SHAPES:-"8192:12 8192:24 8192:48 8192:96 16384:12 16384:24 16384:48 16384:96 32768:12 32768:24 32768:48 32768:96 65536:12 65536:24 65536:48 65536:96"}
KRA=/public/home/tanbo/kra
mkdir -p "$OUT/dev" "$OUT/dev_prev"; rm -f "$OUT/wall.jsonl"
export PYTHONPATH=/public/home/tanbo/xplat/bwenv:$TREE:/public/home/tanbo/codex_validation/stage_g2_bw7/takeover
export KDA_G2=1 TRITON_CACHE_DIR=/tmp/triton_round PYTHONDONTWRITEBYTECODE=1
# device time: shapes dealt over GPUs largest-first (64K/96 dominates), each GPU sequential
i=0; declare -A Q
ORDER=$(for s in $SHAPES; do echo "$(( ${s%%:*} * ${s##*:} )) $s"; done | sort -rn | awk '{print $2}')
for s in $ORDER; do g=$(echo $GPUS | cut -d' ' -f$((i % $(echo $GPUS | wc -w) + 1))); Q[$g]="${Q[$g]} $s"; i=$((i + 1)); done
devrun() {  # tree outdir T H r gpu
  local tag="g2_${3}_${4}_r${5}"; local tmp=/tmp/rva_$(basename $2)_$tag; rm -f $tmp.*
  KRA_JITTER_SEED=$5 HIP_VISIBLE_DEVICES=$6 PYTHONPATH=/public/home/tanbo/xplat/bwenv:$1:/public/home/tanbo/codex_validation/stage_g2_bw7/takeover \
    hipprof --hip-trace --output-type 0 -o $tmp python3 $KRA/examples/kda/kda_iter.py --tree "$1" --arm g2 --T "$3" --H "$4" --iters 5 > /dev/null 2>&1
  cp $tmp.json "$2/xpd_${tag}.json" 2>/dev/null; rm -f $tmp.*
}
for g in "${!Q[@]}"; do
  (
    for s in ${Q[$g]}; do
      T=${s%%:*}; H=${s##*:}
      for r in $(seq 1 "$N"); do
        if [ -n "$PREV" ] && (( r % 2 == 0 )); then devrun "$TREE" "$OUT/dev" $T $H $r $g; devrun "$PREV" "$OUT/dev_prev" $T $H $r $g
        else [ -n "$PREV" ] && devrun "$PREV" "$OUT/dev_prev" $T $H $r $g; devrun "$TREE" "$OUT/dev" $T $H $r $g; fi
      done
    done
  ) &
done
# wall time: sequential on one GPU (host contention would inflate it), concurrent with the above
(
  cd "$TREE"
  for s in $SHAPES; do
    T=${s%%:*}; H=${s##*:}
    for r in $(seq 1 "$NW"); do
      HIP_VISIBLE_DEVICES=$WALL_GPU python3 tests/megatron/kda_op_bench_tp.py "$T" "$H" g2 7 2>/dev/null \
        | grep '"tag": "kda_op_bench"' | sed "s/^/{\"run\": $r, \"host\": \"bw52\", \"data\": /; s/\$/}/" >> "$OUT/wall.jsonl"
    done
  done
) &
wait
cd $KRA && PYTHONPATH=$KRA python3 examples/kda/xplat_bw_dev_parse.py "$OUT/dev" > "$OUT/dev.json"
[ -n "$PREV" ] && PYTHONPATH=$KRA python3 examples/kda/xplat_bw_dev_parse.py "$OUT/dev_prev" > "$OUT/dev_prev.json"
echo ROUND_VS_A800_DONE
