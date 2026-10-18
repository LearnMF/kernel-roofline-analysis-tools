#!/bin/bash
# SQTT of one production chunk-parallel kernel on bw52 + wait-site attribution: cap_k.sh KERNEL TAG [GPU]
KN=$1; TAG=$2; G=${3:-5}; T=${4:-/opt/kda_env/g2_r5}; TT=${5:-16384}; HH=${6:-12}; K=/public/home/tanbo/kra/examples/kda
L=/public/home/tanbo/codex_validation/locks; exec 9>$L/hcu_$G.lock; flock -w 900 9 || { echo LOCK_BUSY; exit 75; }
export KDA_G2=1 HIP_VISIBLE_DEVICES=$G
D=/tmp/sqk_$TAG; rm -rf $D; mkdir -p $D
cd /tmp && PYTHONPATH=/public/home/tanbo/xplat/bwenv:$T:/public/home/tanbo/codex_validation/stage_g2_bw7/takeover hipprof --sqtt --sqtt-type 1 --kernel-name $KN --output-type 0 -o $D/x \
   python3 $K/kda_iter.py --tree $T --T $TT --H $HH --iters 1 --warmup 1 > $D/log.txt 2>&1
echo rc=$?; ls $D | grep -oE "thread_trace_[0-9]+" | sort | uniq -c
O=/public/home/tanbo/xplat/sqk; mkdir -p $O
python3 $K/sqtt_hotspots.py $D --dispatch 0 --top 30 --json $O/$TAG.json > $O/$TAG.txt 2>&1; head -70 $O/$TAG.txt
