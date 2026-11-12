#!/bin/bash
# SQTT of every production KDA kernel on gfx938: cap_chain.sh T H
T=$1; H=$2
source /opt/dtk/env.sh >/dev/null 2>&1
export HIP_VISIBLE_DEVICES=6 PYTHONPATH=/public/home/tanbo/ft938_a83
cd /tmp
for K in g2_gate_kernel g2_fwd_prep_a_kernel g2_fwd_prep_b_kernel g2_fwd_h_v8o4 g2_wu_kernel kda_gn_all_kernel g2_dav_kernel g2_dhu_v8b p3_wy_s4a_kernel p3_wy_s4b_kernel g2_bwd_tail_kernel; do
  D=/public/home/tanbo/sqtt938/chain_${T}_${H}/$K; rm -rf $D; mkdir -p $D
  hipprof --sqtt --sqtt-type 1 --kernel-name $K --output-type 0 -o $D/x \
     python3 /public/home/tanbo/mk3/ft_card_run.py --T $T --H $H --iters 1 --warmup 1 > $D/log.txt 2>&1
  echo "$K rc=$? files=$(ls $D | grep -c thread_trace)"
done
