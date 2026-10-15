#!/bin/bash
K=/public/home/tanbo/kra/examples/kda; T=/opt/kda_env/g2_p70b; L=/public/home/tanbo/codex_validation/locks/bw46
exec 9>$L/hcu_6.lock; flock -w 900 9 || exit 75
export KDA_G2=1 HIP_VISIBLE_DEVICES=6 PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover KDA_G2_PIPE=1 KDA_G2_PIPE_K=48 KDA_G2_PIPE_MASK=1
D=/tmp/p70tr; rm -rf $D; mkdir -p $D; cd $D
hipprof --hip-trace --output-type 0 -o $D/x python3 $K/kda_iter.py --tree $T --T 16384 --H 12 --iters 2 --warmup 1 > log.txt 2>&1
ls $D | head
f=$(ls $D/*hipkernel*.csv 2>/dev/null | head -1); [ -n "$f" ] || f=$(ls $D/*.csv | head -1)
head -2 "$f"
