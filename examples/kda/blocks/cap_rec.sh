#!/bin/bash
# SQTT of the production recurrence g2_fwd_h_v8o4_kernel at 16K/12 (TP8): fwd MODE 1 + recompute MODE 4
L=/public/home/tanbo/codex_validation/locks; exec 9>$L/hcu_4.lock; flock -n 9 || { echo LOCK_BUSY; exit 75; }
hy-smi -d 4 --showmemuse 2>&1 | grep -q 'HCU memory use (%): 0' || { echo HBM_BUSY; exit 76; }
export KDA_G2=1 HIP_VISIBLE_DEVICES=4
PP=/public/home/tanbo/xplat/bwenv:/public/home/tanbo/g2_opt:/public/home/tanbo/codex_validation/stage_g2_bw7/takeover
D=/tmp/sqrec_v8o4; rm -rf $D; mkdir -p $D
cd /tmp && PYTHONPATH=$PP hipprof --sqtt --sqtt-type 1 --kernel-name g2_fwd_h_v8o4_kernel --output-type 0 -o $D/x \
   python3 /public/home/tanbo/kra/examples/kda/kda_iter.py --tree /public/home/tanbo/g2_opt --T 16384 --H 12 --iters 1 --warmup 1 > $D/log.txt 2>&1
echo rc=$?; ls $D | grep -oE "thread_trace_[0-9]+" | sort | uniq -c
