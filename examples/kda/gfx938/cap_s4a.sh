#!/bin/bash
# build ba078ed and ba078ed + P-938h into /public, then SQTT s4a for both (run in the sqtt container)
source /opt/dtk/env.sh >/dev/null 2>&1; export DTKROOT=/opt/dtk
R=/public/home/tanbo
for v in ba0 h6; do
  rm -rf /tmp/src_$v $R/ft938_$v && cp -a $R/src_ba0 /tmp/src_$v
  [ $v = h6 ] && (cd /tmp/src_$v && patch -p1 < $R/mk3/p938h_s4a_mls_ring.diff > /dev/null && echo patched)
  (cd /tmp/src_$v && python3 -m pip install . --no-deps --target $R/ft938_$v > /tmp/bld_$v.log 2>&1; echo "build $v rc=$?")
done
cd /tmp
for v in ba0 h6; do
  D=$R/sqtt938/s4a_$v; rm -rf $D; mkdir -p $D
  HIP_VISIBLE_DEVICES=6 PYTHONPATH=$R/ft938_$v hipprof --sqtt --sqtt-type 1 --kernel-name p3_wy_s4a_kernel --output-type 0 -o $D/x \
     python3 $R/mk3/ft_card_run.py --T 2048 --H 16 --iters 1 --warmup 1 > $D/log.txt 2>&1
  echo "sqtt $v rc=$? files=$(ls $D | grep -c thread_trace)"
done
