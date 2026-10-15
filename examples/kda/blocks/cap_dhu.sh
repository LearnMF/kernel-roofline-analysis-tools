#!/bin/bash
# SQTT of the production recurrence (bw7 g2_r5) at 16K/12 + critical-path attribution: cap_b7.sh TREE TAG
T=${1:-/opt/kda_env/g2_r5}; TAG=${2:-prod}
L=/public/home/tanbo/codex_validation/locks/bw7; exec 9>$L/hcu_6.lock; flock -w 900 9 || { echo LOCK_BUSY; exit 75; }
hy-smi -d 6 --showmemuse 2>&1 | grep -q 'HCU memory use (%): 0' || { echo HBM_BUSY; exit 76; }
export KDA_G2=1 HIP_VISIBLE_DEVICES=6
D=/tmp/sqdhu_$TAG; rm -rf $D; mkdir -p $D
cd /tmp && PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover hipprof --sqtt --sqtt-type 1 --kernel-name g2_dhu_v8b_kernel --output-type 0 -o $D/x \
   python3 /public/home/tanbo/kra/examples/kda/kda_iter.py --tree $T --T 16384 --H 12 --iters 1 --warmup 1 > $D/log.txt 2>&1
echo rc=$?; ls $D | grep -oE "thread_trace_[0-9]+" | sort | uniq -c
O=/public/home/tanbo/xplat/cp; mkdir -p $O
for d in 1; do python3 /public/home/tanbo/kra/examples/kda/blocks/cp_step.py $D --dispatch $d --top 40 --json $O/cpdhu_${TAG}_d$d.json > $O/cpdhu_${TAG}_d$d.txt 2>&1; head -32 $O/cpdhu_${TAG}_d$d.txt; done
