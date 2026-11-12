#!/bin/bash
# lb_attr.sh VAR GPU SHAPE OUT MODE : per-phase LDS counters; MODE only = skip all but phase k, skip = skip phase k
V=$1; G=$2; SH=$3; OUT=$4; MODE=$5; M=/public/home/tanbo/mk3; rm -f $OUT $OUT.done
run() { bash $M/lb_pmc.sh $V $G $SH "$1" KDA_G2_DAV_LSKIP=$2 KDA_G2_WU_LSKIP=$3 >> $OUT; }
if [ $MODE = only ]; then
  run full 0 0; run none 255 1023
  for k in 0 1 2 3 4 5 6 7 8 9; do d=$(( k < 8 ? 255 ^ (1 << k) : 255 )); w=$(( 1023 ^ (1 << k) )); run only$k $d $w; done
else
  for k in 0 1 2 3 4 5 6 7 8 9; do d=$(( k < 8 ? 1 << k : 0 )); w=$(( 1 << k )); run skip$k $d $w; done
fi
echo DONE > $OUT.done
