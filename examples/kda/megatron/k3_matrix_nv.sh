#!/usr/bin/env bash
# A800 K3-slice Megatron matrix (same protocol as codex_validation/runs/megatron/k3_matrix_r5.sh:
# 12 iters, step time = median of iters 4-9, rank0 torch-profiler trace of steps 10-11, K3_MEM log).
# 4 GPUs (EP4), L4 slice (3 KDA + 1 MLA): TP4 8K (recompute on/off), TP4 16K/32K and TP2 8K recompute.
#   bash k3_matrix_nv.sh [TAG=nvmx]       (inside the NGC container)
TAG=${1:-nvmx}
F=/public/home/tanbo/xplat/k3_slice_train_nv.sh
OUTD=/public/home/tanbo/xplat/megatron; export OUTD K3_MEM_LOG=1
export SLICE=/public/home/tanbo/codex_validation/runs/megatron/hf/Kimi-K3-slice-L4-E16 NL=4   # 3 KDA + 1 MLA (L8 OOMs on 4 GPUs)
RC="--recompute-granularity full --recompute-method uniform --recompute-num-layers 1"
PROF="--profile --use-pytorch-profiler --profile-step-start 10 --profile-step-end 11 --profile-ranks 0"

settle() {  # never start a run while a previous one still holds the GPUs (killed runs leave workers)
  pkill -TERM -f pretrain_gpt.py 2>/dev/null; sleep 5
  for i in $(seq 1 60); do
    n=$(pgrep -f pretrain_gpt.py | wc -l); [ "$n" -eq 0 ] && return 0; sleep 5
  done
  echo "WARN: pretrain_gpt still resident"; }
run() {  # seq tp recompute(0/1) arm
  local seq=$1 tp=$2 rc=$3 arm=$4
  local name=${TAG}_${arm}_r${rc}
  local d=$OUTD/prof/${name}_s${seq}_tp${tp}; rm -rf "$d"; mkdir -p "$d"; cd "$d" || return
  if [ "$arm" = flanr ]; then export KDA_FLA_DISABLE_RECOMPUTE=1; else unset KDA_FLA_DISABLE_RECOMPUTE; fi
  if [ "$rc" = 1 ]; then export RECOMP_ARGS="$RC"; else unset RECOMP_ARGS; fi
  export PROF_ARGS="$PROF" PORT=$((29700 + RANDOM % 200))
  settle
  echo "=== $(date +%T) $name seq=$seq tp=$tp"
  timeout 3600 bash $F "$seq" "$tp" 12 "$name"
}
for arm in fla flanr; do
  run 8192 4 1 $arm
  run 8192 4 0 $arm
  run 16384 4 1 $arm
  run 8192 2 1 $arm
  run 32768 4 1 $arm
done
echo NV_MATRIX_DONE
