#!/bin/bash
# abv.sh GPU SHAPES OUT "tag=build[:ENV=V,ENV2=V]" ... : per-kernel times (kbench --all, mean of 20 iters) of several
# variants, order v1..vn vn..v1 twice (each variant 4 fresh processes), warm-up first.
source /opt/dtk/env.sh >/dev/null 2>&1; cd /tmp; G=$1; SH=$2; OUT=$3; shift 3
M=/public/home/tanbo/mk3; B=/public/home/tanbo/b938; rm -f $OUT $OUT.done
run() { spec=$1; tag=${spec%%=*}; rest=${spec#*=}; bld=${rest%%:*}; envs=""; [ "$rest" != "$bld" ] && envs=${rest#*:}
  env FTRAIN_KDA_DEV=1 ${envs//,/ } HIP_VISIBLE_DEVICES=$G PYTHONPATH=$B/$bld python3 $M/kbench.py --all --shapes $SH --iters 20 --tag $tag 2>/dev/null | grep '^{' >> $2; }
run "$1" /dev/null
SP=("$@"); N=${#SP[@]}
for r in 1 2; do for ((i=0;i<N;i++)); do run "${SP[$i]}" $OUT; done; for ((i=N-1;i>=0;i--)); do run "${SP[$i]}" $OUT; done; done
echo DONE > $OUT.done
