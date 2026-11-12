#!/bin/bash
# after-run: ft only, Megatron-LM-k3 call.  usage: mk3_run2.sh GPU "T list" OUT
source /opt/dtk/env.sh >/dev/null 2>&1
G=$1; TS=$2; OUT=$3; B=/public4/home/tanbo/mk3/mk3_bench.py
cd /tmp
for T in $TS; do for H in 64 32 16 8; do
  HIP_VISIBLE_DEVICES=$G python3 $B $T $H ft 7 mk3 2>/dev/null | grep '^{' >> $OUT
done; done
echo DONE >> $OUT.done
