#!/bin/bash
# 16-shape operator table, H20 side (tanbo_kda_h20: torch 2.10.0+cu128 / triton 3.6.0 / FLA 0.5.2 in
# /home/locauser/kda_site, python -S so the image site-packages stay out) -- same protocol as A800
# op16_nv.sh: per shape, sequential on one GPU, N processes of kda_op_bench_nv.py (fla, fla_nr).
GPU=$1; OUT=$2; N=${3:-3}; SHAPES=$4
export PYTHONPATH=/home/locauser/kda_site
for s in $SHAPES; do
  T=${s%%:*}; H=${s##*:}
  for r in $(seq 1 "$N"); do
    CUDA_VISIBLE_DEVICES=$GPU python3 -S /home/locauser/kda_op_bench_nv.py "$T" "$H" fla,fla_nr 7 2>/dev/null \
      | grep "\"tag\": \"kda_op_bench\"" | sed "s/^/{\"run\": $r, \"host\": \"h20\", \"data\": /; s/\$/}/" >> "$OUT"
  done
  echo "$(date +%T) done $s"
done
echo OP16_H20_DONE
