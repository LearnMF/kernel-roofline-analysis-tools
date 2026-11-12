#!/bin/bash
source /opt/dtk/env.sh >/dev/null 2>&1; export DTKROOT=/opt/dtk; R=/public/home/tanbo
rm -rf /tmp/src_hd $R/fthd_s && cp -a $R/ft_938_src /tmp/src_hd
(cd /tmp/src_hd && python3 -m pip install . --no-deps --target $R/fthd_s > /tmp/bhd.log 2>&1; echo "build rc=$?")
cd /tmp
for sh in "8192 64" "2048 16"; do set -- $sh; D=$R/sqtt938/s4b_hd_$1_$2; rm -rf $D; mkdir -p $D
  HIP_VISIBLE_DEVICES=6 PYTHONPATH=$R/fthd_s hipprof --sqtt --sqtt-type 1 --kernel-name p3_wy_s4b_kernel --output-type 0 -o $D/x \
     python3 $R/mk3/ft_card_run.py --T $1 --H $2 --iters 1 --warmup 1 > $D/log.txt 2>&1
  echo "sqtt $1/$2 rc=$? files=$(ls $D | grep -c thread_trace) size=$(du -sh $D | cut -f1)"
done
