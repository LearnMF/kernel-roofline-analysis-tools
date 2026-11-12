#!/bin/bash
# SQTT: fused prep_ab (R-only fusion, vR) vs the pair prep_a / prep_b, 2048/16, gfx938
source /opt/dtk/env.sh >/dev/null 2>&1; export DTKROOT=/opt/dtk; R=/public/home/tanbo
rm -rf /tmp/src_vR $R/ftvR_s && cp -a $R/ft_938_src /tmp/src_vR && cp $R/mk3/prep_vR.cuh /tmp/src_vR/src/primitive/hygon/kda/g2/g2_fwd_prep.cuh
(cd /tmp/src_vR && python3 -m pip install . --no-deps --target $R/ftvR_s > /tmp/bvr.log 2>&1; echo "build rc=$?")
cd /tmp
for spec in "fused:g2_fwd_prep_ab_kernel:1" "pa:g2_fwd_prep_a_kernel:0" "pb:g2_fwd_prep_b_kernel:0"; do
  IFS=: read nm kn fu <<< "$spec"; D=$R/sqtt938/fu_$nm; rm -rf $D; mkdir -p $D
  FTRAIN_KDA_DEV=1 KDA_G2_PREP_FUSE=$fu HIP_VISIBLE_DEVICES=6 PYTHONPATH=$R/ftvR_s hipprof --sqtt --sqtt-type 1 --kernel-name $kn --output-type 0 -o $D/x \
     python3 $R/mk3/ft_card_run.py --T 2048 --H 16 --iters 1 --warmup 1 > $D/log.txt 2>&1
  echo "sqtt $nm rc=$? files=$(ls $D | grep -c thread_trace)"
done
