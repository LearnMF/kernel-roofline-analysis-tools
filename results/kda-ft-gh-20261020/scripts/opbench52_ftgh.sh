#!/bin/bash
# FT (flash-train feat/kda + P-77b/P-82 ports, md5 4d01bebe5ba8) vs hip_kda (g2_opt P-88) inside tanbo_mega_bw52: kda_op_bench_tp.py arms g2,ft,
# same process + same inputs, interleaved; BENCH_CKPT=1 adds the Megatron full-recompute step (reentrant checkpoint).
dev=$1; SHAPES=$2; N=${3:-2}; OUT=${4:-/public/home/tanbo/xplat/opbench52_ftgh_vs_g2.jsonl}
exec 9>/public/home/tanbo/codex_validation/locks/hcu_${dev}.lock; flock -w 900 9 || { echo LOCK_BUSY; exit 75; }
hy-smi -d $dev --showuse 2>&1 | grep -q "HCU use (%): 0.0" || { echo HCU_BUSY; exit 76; }
export HIP_VISIBLE_DEVICES=$dev KDA_G2=1 BENCH_CKPT=1 TRITON_CACHE_DIR=/tmp/triton_ft_$dev PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${FT_SITE:-/public/home/tanbo/xplat/ft_site_gh}:/public/home/tanbo/xplat/bwenv:/public/home/tanbo/g2_opt:/public/home/tanbo/codex_validation/stage_g2_bw7/takeover
cd /public/home/tanbo/g2_opt
for s in $SHAPES; do T=${s%%:*}; H=${s##*:}
  for r in $(seq 1 $N); do
    line=$(timeout 2400 python3 /public/home/tanbo/xplat/kda_op_bench_ftgh.py $T $H g2,ft,ftc 7 2>/dev/null | grep "\"tag\": \"kda_op_bench\"")
    [ -n "$line" ] && echo "{\"run\": $r, \"host\": \"bw52\", \"hcu\": $dev, \"data\": $line}" >> $OUT || echo "FAIL $s run $r"
  done; echo "done $s $(date +%T)"
done; echo STREAM_DONE_$dev
