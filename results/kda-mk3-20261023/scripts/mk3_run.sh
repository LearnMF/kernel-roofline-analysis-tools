#!/bin/bash
# baseline: Yu Hui sizes, Megatron-LM-k3 call (mk3) for ft + fla, and the K3 call (k3) for ft (route comparison)
# usage: mk3_run.sh GPU "T list" OUT
source /opt/dtk/env.sh >/dev/null 2>&1
G=$1; TS=$2; OUT=$3; B=/public4/home/tanbo/mk3/mk3_bench.py
cd /tmp
for T in $TS; do for H in 64 32 16 8; do
  HIP_VISIBLE_DEVICES=$G python3 $B $T $H ft,fla 7 mk3 2>/dev/null | grep '^{' >> $OUT
  HIP_VISIBLE_DEVICES=$G python3 $B $T $H ft 7 k3 2>/dev/null | grep '^{' >> $OUT
done; done
echo DONE >> $OUT.done
