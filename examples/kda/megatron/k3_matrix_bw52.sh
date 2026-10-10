#!/usr/bin/env bash
# BW (bw52) counterpart of k3_matrix_nv.sh: the identical 4-GPU K3-slice configs (EP4, L4 slice = 3 KDA + 1 MLA,
# 12 iters, rank0 profiler trace of steps 10-11).  Arms: fla (FLA 0.5.2 + DCU autotune pruning),
# flanr (FLA disable_recompute), hip0 (hip_kda G2, KDA_G2_KEEP_STATE=0: no memory-for-speed trade).
#   bash k3_matrix_bw52.sh [TAG=bwmx]      (inside tanbo_mega_bw52)
TAG=${1:-bwmx}
M=/public/home/tanbo/codex_validation/runs/megatron; D=$M/Megatron-LM-das
OUTD=/public/home/tanbo/xplat/megatron; mkdir -p "$OUTD/logs"
RC="--recompute-granularity full --recompute-method uniform --recompute-num-layers 1"
PROF="--profile --use-pytorch-profiler --profile-step-start 10 --profile-step-end 11 --profile-ranks 0"
export PYTHONPATH=/public/home/tanbo/xplat/bwenv:$D:$D/3rdparty/Megatron-LM:$M/Megatron-Bridge/src:/public/home/tanbo/g2_opt:/public/home/tanbo/codex_validation/stage_g2_bw7/takeover
export KDA_G2=1 CUDA_DEVICE_MAX_CONNECTIONS=1 GPU_MAX_HW_QUEUES=4 PYTORCH_ALLOC_CONF=expandable_segments:True
export KDA_TRITON_CACHE_DIR=/tmp/kda_triton_mega TRITON_CACHE_DIR=/tmp/triton_mega K3_MEM_LOG=1
export HIP_VISIBLE_DEVICES=${GPUS:-0,1,2,3}

settle() {  # never start a run while a previous one still holds the GPUs (killed runs leave workers)
  pkill -TERM -f pretrain_gpt.py 2>/dev/null; sleep 5
  for i in $(seq 1 60); do
    n=$(pgrep -f pretrain_gpt.py | wc -l); [ "$n" -eq 0 ] && return 0; sleep 5
  done
  echo "WARN: pretrain_gpt still resident"; }
run() {  # seq tp recompute arm
  local seq=$1 tp=$2 rc=$3 arm=$4 name=${TAG}_${4}_r${3}
  local d=$OUTD/prof/${name}_s${seq}_tp${tp}; rm -rf "$d"; mkdir -p "$d"; cd "$d" || return
  unset KDA_FLA_DISABLE_RECOMPUTE KDA_G2_KEEP_STATE
  case $arm in fla) export KDA_K3_BACKEND=fla;; flanr) export KDA_K3_BACKEND=fla KDA_FLA_DISABLE_RECOMPUTE=1;;
               hip0) export KDA_K3_BACKEND=hip KDA_G2_KEEP_STATE=0;; esac
  local ra=""; [ "$rc" = 1 ] && ra="$RC"
  settle
  echo "=== $(date +%T) $name seq=$seq tp=$tp"
  timeout 3600 torchrun --nproc_per_node 4 --master_port $((29700 + RANDOM % 200)) $D/pretrain_gpt.py \
    --use-bridge --bridge-hf-model $M/hf/${SLICE_NAME:-Kimi-K3-slice-L4-E16} \
    --tensor-model-parallel-size "$tp" --pipeline-model-parallel-size 1 --context-parallel-size 1 \
    --sequence-parallel --num-experts 16 --moe-router-topk 16 --moe-ffn-hidden-size 3072 --expert-model-parallel-size 4 --expert-tensor-parallel-size 1 \
    --num-layers ${NL:-4} --hidden-size 7168 --num-attention-heads 96 --kv-channels 128 \
    --seq-length "$seq" --max-position-embeddings "$seq" --position-embedding-type none \
    --micro-batch-size 1 --global-batch-size $(( 4 / tp )) --train-iters 12 \
    --lr 1e-5 --min-lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --clip-grad 1.0 \
    --bf16 --transformer-impl transformer_engine --use-mcore-models \
    --mock-data --tokenizer-type NullTokenizer --vocab-size 163840 \
    --use-distributed-optimizer --seed 1234 --init-method-std 0.006 \
    --log-interval 1 --save-interval 1000000 --eval-iters 0 --eval-interval 1000000 --log-throughput \
    --log-memory-to-tensorboard $ra $PROF > "$OUTD/logs/${name}_s${seq}_tp${tp}.log" 2>&1
  echo "rc=$?"
}
for arm in ${ARMS:-hip0 fla flanr}; do
  run 8192 4 1 $arm
  run 8192 4 0 $arm
  run 16384 4 1 $arm
  run 8192 2 1 $arm
  run 32768 4 1 $arm
done
echo BW_MATRIX_DONE
