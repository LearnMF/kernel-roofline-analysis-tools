#!/bin/bash
K=/public/home/tanbo/kra/examples/kda; T=/opt/kda_env/g2_r5; L=/public/home/tanbo/codex_validation/locks/bw46; G=${1:-1}
exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75
hy-smi -d $G --showmemuse 2>&1 | grep -q 'HCU memory use (%): 0' || { echo HBM_BUSY; exit 76; }
export KDA_G2=1 HIP_VISIBLE_DEVICES=$G PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover; cd /tmp
for TH in "16384 24" "32768 24" "65536 24" "8192 48" "65536 48" "16384 32"; do set -- $TH; echo "== dhu T=$1 H=$2"
  python3 $K/op_env_time.py --tree $T --fn dhu_bn --T $1 --H $2 --env KDA_G2_R1A_DHU --vals 1,0 --reps 10 2>&1 | grep -E " us |BITWISE|DIFF|rror"
done
echo P85A_DONE
