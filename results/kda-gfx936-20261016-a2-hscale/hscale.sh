#!/bin/bash
# dhu H-scaling at fixed T=16384 (same v8b<1,true> instance forced): per-CU bound -> flat up to 80 CTAs; HBM bound -> ~linear in H
K=/public/home/tanbo/kra/examples/kda; T=/opt/kda_env/g2_r5; L=/public/home/tanbo/codex_validation/locks/bw46; G=${1:-6}
exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75
hy-smi -d $G --showmemuse 2>&1 | grep -q 'HCU memory use (%): 0' || { echo HBM_BUSY; exit 76; }
export KDA_G2=1 HIP_VISIBLE_DEVICES=$G KDA_G2_R1A_DHU=0 KDA_G2_DHU_LIN=1 PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover
for H in 12 16 20 24 32 40; do
  D=/tmp/hs_$H; rm -rf $D; mkdir -p $D; cd $D
  hipprof --hip-trace --output-type 0 -o $D/x python3 $K/kda_iter.py --tree $T --T 16384 --H $H --iters 3 --warmup 1 > log.txt 2>&1
  f=$(ls $D/*hipkernel*.csv | head -1)
  echo "H=$H ctas=$((4*H)): $(grep -E 'dhu_v8b_kernel|fwd_h_v8o4' $f | awk -F'","' '{gsub(/"/,"",$1); gsub(/"/,"",$4); printf "%s=%.1fus  ", substr($1,1,40), $4/1000}')"
done
echo HS_DONE
