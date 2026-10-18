#!/usr/bin/env bash
# opbench46.sh <hcu> "<T:H ...>" [N=3] [OUT] -- run INSIDE tanbo_kda_b46 (megatron-0.18.2 dtk2604 image, same as the
# Feishu KDA doc): tests/megatron/kda_op_bench_tp.py arms fla,fla_nr,g2 (same process, same inputs, interleaved), N processes.
dev=$1; SHAPES=$2; N=${3:-3}; OUT=${4:-/public/home/tanbo/xplat/opbench46_p88.jsonl}
L=/public/home/tanbo/codex_validation/locks/bw46
exec 9>"$L/hcu_${dev}.lock"; flock -w 900 9 || { echo LOCK_BUSY; exit 75; }
hy-smi -d $dev --showuse 2>&1 | grep -q "HCU use (%): 0.0" || { echo HCU_BUSY; exit 76; }
hy-smi -d $dev --showmemuse 2>&1 | grep -q "HCU memory use (%): [0-5]$" || { echo HBM_BUSY; exit 77; }
export HIP_VISIBLE_DEVICES=$dev KDA_G2=1 TRITON_CACHE_DIR=/tmp/triton_mega_$dev PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/opt/kda_env:/opt/kda_env/g2_r5:/opt/kda_env/takeover
cd /opt/kda_env/g2_r5
for s in $SHAPES; do T=${s%%:*}; H=${s##*:}
  for r in $(seq 1 $N); do
    line=$(timeout 2400 python3 tests/megatron/kda_op_bench_tp.py $T $H fla,fla_nr,g2 7 2>/dev/null | grep "\"tag\": \"kda_op_bench\"")
    [ -n "$line" ] && echo "{\"run\": $r, \"host\": \"bw46\", \"hcu\": $dev, \"data\": $line}" >> $OUT || echo "FAIL $s run $r"
  done
  echo "done $s $(date +%T)"
done
echo STREAM_DONE_$dev
