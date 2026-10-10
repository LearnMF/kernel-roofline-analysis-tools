#!/bin/bash
# 16-shape operator table, BW side (inside tanbo_mega_bw52): T in {8K,16K,32K,64K} x H in {96,48,24,12}.
# Per shape, SEQUENTIAL on one GPU: N wall-time processes of the tree's kda_op_bench_tp.py (arms
# fla = FLA 0.5.2 default + DCU autotune pruning, fla_nr = disable_recompute, g2 = hip_kda; CUDA
# events, median of 7, same protocol as the BW report), then N hipprof device-time processes per arm.
#   bash op16_bw.sh GPU OUTDIR [N=3] [SHAPES]
GPU=$1; OUT=$2; N=${3:-3}
SHAPES=${4:-"8192:12 8192:24 8192:48 8192:96 16384:12 16384:24 16384:48 16384:96 32768:12 32768:24 32768:48 32768:96 65536:12 65536:24 65536:48 65536:96"}
mkdir -p "$OUT/dev"
export PYTHONPATH=/public/home/tanbo/xplat/bwenv:/public/home/tanbo/g2_opt:/public/home/tanbo/codex_validation/stage_g2_bw7/takeover
export HIP_VISIBLE_DEVICES=$GPU KDA_G2=1 TRITON_CACHE_DIR=/tmp/triton_op16 PYTHONDONTWRITEBYTECODE=1
cd /public/home/tanbo/g2_opt
for s in $SHAPES; do
  T=${s%%:*}; H=${s##*:}
  for r in $(seq 1 "$N"); do
    python3 tests/megatron/kda_op_bench_tp.py "$T" "$H" fla,fla_nr,g2 7 2>/dev/null | grep '"tag": "kda_op_bench"' \
      | sed "s/^/{\"run\": $r, \"host\": \"bw52\", \"data\": /; s/\$/}/" >> "$OUT/wall.jsonl"
  done
  for arm in g2 fla fla_nr; do
    for r in $(seq 1 "$N"); do
      tag="${arm}_${T}_${H}_r${r}"
      rm -f /tmp/op16_$tag.*
      hipprof --hip-trace --output-type 0 -o /tmp/op16_$tag python3 /public/home/tanbo/kra/examples/kda/kda_iter.py \
        --tree /public/home/tanbo/g2_opt --arm "$arm" --T "$T" --H "$H" --iters 5 > /dev/null 2>&1
      cp /tmp/op16_$tag.json "$OUT/dev/xpd_${tag}.json" 2>/dev/null; rm -f /tmp/op16_$tag.*
    done
  done
  echo "$(date +%T) done $s"
done
echo OP16_BW_DONE
