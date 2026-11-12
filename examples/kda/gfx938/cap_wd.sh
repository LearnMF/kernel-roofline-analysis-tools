#!/bin/bash
source /opt/dtk/env.sh >/dev/null 2>&1; R=/public/home/tanbo; cd /tmp
for spec in "g2_wu_kernel 8192 64" "g2_dhu_v8b_kernel 8192 64" "g2_dhu_v8b_kernel 16384 8"; do set -- $spec
  D=$R/sqtt938/${1}_$2_$3; rm -rf $D; mkdir -p $D
  HIP_VISIBLE_DEVICES=6 PYTHONPATH=$R/fthd_s hipprof --sqtt --sqtt-type 1 --kernel-name $1 --output-type 0 -o $D/x \
     python3 $R/mk3/ft_card_run.py --T $2 --H $3 --iters 1 --warmup 1 > $D/log.txt 2>&1
  echo "sqtt $1 $2/$3 rc=$? files=$(ls $D | grep -c thread_trace) size=$(du -sh $D | cut -f1)"
  python3 $R/mk3/stall_ctx.py $D 4 22
done
