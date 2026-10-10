#!/usr/bin/env bash
# NVIDIA counterpart of codex_validation/runs/megatron/k3_slice_train_r5.sh: the SAME code
# (Megatron-LM-das core 0.18.2 + Megatron-Bridge K3 model), slice config and arguments, with the
# KDA backend = FLA chunk_kda.  Run inside the NGC container (shared /public mount).
#   k3_slice_train_nv.sh <seq_len> <tp> <iters> <tag>      env: NPROC (4), EP (4), NL (8), SLICE, RECOMP_ARGS, PROF_ARGS
set -u
SEQ=$1; TP=$2; ITERS=$3; TAG=$4
M=/public/home/tanbo/codex_validation/runs/megatron; D=$M/Megatron-LM-das
SLICE=${SLICE:-$M/hf/Kimi-K3-slice-L8-E16}
OUTD=${OUTD:-/public/home/tanbo/xplat/megatron}; mkdir -p "$OUTD/logs"
LOG=$OUTD/logs/${TAG}_s${SEQ}_tp${TP}.log
export PYTHONPATH=$D:$D/3rdparty/Megatron-LM:$M/Megatron-Bridge/src
export KDA_K3_BACKEND=fla CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
NPROC=${NPROC:-4}
torchrun --nproc_per_node "$NPROC" --master_port "${PORT:-29611}" $D/pretrain_gpt.py \
  --use-bridge --bridge-hf-model "$SLICE" \
  --tensor-model-parallel-size "$TP" --pipeline-model-parallel-size 1 --context-parallel-size 1 \
  --sequence-parallel --num-experts 16 --moe-router-topk 16 --moe-ffn-hidden-size 3072 --expert-model-parallel-size "${EP:-4}" --expert-tensor-parallel-size 1 \
  --num-layers "${NL:-8}" --hidden-size 7168 --num-attention-heads 96 --kv-channels 128 \
  --seq-length "$SEQ" --max-position-embeddings "$SEQ" --position-embedding-type none \
  --micro-batch-size 1 --global-batch-size $(( NPROC / TP )) --train-iters "$ITERS" \
  --lr 1e-5 --min-lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --clip-grad 1.0 \
  --bf16 --transformer-impl transformer_engine --use-mcore-models \
  --mock-data --tokenizer-type NullTokenizer --vocab-size 163840 \
  --use-distributed-optimizer --seed 1234 --init-method-std 0.006 \
  --log-interval 1 --save-interval 1000000 --eval-iters 0 --eval-interval 1000000 --log-throughput \
  --log-memory-to-tensorboard ${RECOMP_ARGS:-} ${PROF_ARGS:-} > "$LOG" 2>&1
echo "rc=$? log=$LOG"
