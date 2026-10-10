#!/bin/bash
# Cross-platform comparison, BW side: the production tree's operator bench (arms fla,g2, Megatron
# checkpoint form on) for the given shapes, N independent processes each.  Run on the bw7 HOST:
#   bash xplat_bw.sh GPU "8192:12 8192:24" [N=3] [OUT=/public/home/tanbo/xplat/bw.jsonl]
GPU=$1; SHAPES=$2; N=${3:-3}; OUT=${4:-/public/home/tanbo/xplat/bw.jsonl}
mkdir -p "$(dirname "$OUT")"
for s in $SHAPES; do
  T=${s%%:*}; H=${s##*:}
  for r in $(seq 1 "$N"); do
    docker exec -e HIP_VISIBLE_DEVICES="$GPU" -e KDA_G2=1 -e BENCH_CKPT=1 -e TRITON_CACHE_DIR=/tmp/triton_mega \
      -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/opt/kda_env:/opt/kda_env/g2_r5:/opt/kda_env/takeover \
      tanbo_mega_k3 bash -lc "cd /opt/kda_env/g2_r5 && python3 tests/megatron/kda_op_bench_tp.py $T $H fla,g2 7 2>/dev/null | grep '\"tag\": \"kda_op_bench\"'" \
      | sed "s/^/{\"run\": $r, \"host\": \"bw7\", \"data\": /; s/\$/}/" >> "$OUT"
  done
done
echo "done $SHAPES"
