#!/bin/bash
# Cross-platform comparison, NVIDIA side: FLA chunk_kda (arms fla, fla_nr; Megatron checkpoint
# form on) for the given shapes, N independent processes each.  Run INSIDE the NV container:
#   bash xplat_nv.sh GPU "8192:12 8192:24" [N=3] [OUT=/xplat/nv.jsonl]
GPU=$1; SHAPES=$2; N=${3:-3}; OUT=${4:-/xplat/nv.jsonl}
for s in $SHAPES; do
  T=${s%%:*}; H=${s##*:}
  for r in $(seq 1 "$N"); do
    CUDA_VISIBLE_DEVICES="$GPU" BENCH_CKPT=1 python /xplat/kda_op_bench_nv.py "$T" "$H" fla,fla_nr 7 2>/dev/null \
      | grep '"tag": "kda_op_bench"' | sed "s/^/{\"run\": $r, \"host\": \"a800\", \"data\": /; s/\$/}/" >> "$OUT"
  done
done
echo "done $SHAPES"
