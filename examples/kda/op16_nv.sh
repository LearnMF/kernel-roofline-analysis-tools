#!/bin/bash
# 16-shape operator table, A800 side (inside tanbo_kda_nv210: torch 2.10 / triton 3.6 / FLA 0.5.2):
# per shape, SEQUENTIAL on one GPU, N processes of kda_op_bench_nv.py (fla, fla_nr; CUDA events
# median of 7 + CUPTI device sum).    bash op16_nv.sh GPU OUT.jsonl [N=3] [SHAPES]
GPU=$1; OUT=$2; N=${3:-3}
SHAPES=${4:-"8192:12 8192:24 8192:48 8192:96 16384:12 16384:24 16384:48 16384:96 32768:12 32768:24 32768:48 32768:96 65536:12 65536:24 65536:48 65536:96"}
for s in $SHAPES; do
  T=${s%%:*}; H=${s##*:}
  for r in $(seq 1 "$N"); do
    CUDA_VISIBLE_DEVICES=$GPU python /xplat/kda_op_bench_nv.py "$T" "$H" fla,fla_nr 7 2>/dev/null \
      | grep '"tag": "kda_op_bench"' | sed "s/^/{\"run\": $r, \"host\": \"a800\", \"data\": /; s/\$/}/" >> "$OUT"
  done
  echo "$(date +%T) done $s"
done
echo OP16_NV_DONE
