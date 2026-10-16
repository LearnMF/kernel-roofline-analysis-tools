#!/bin/bash
K=/public/home/tanbo/kra/examples/kda; T=/opt/kda_env/g2_p80a; L=/public/home/tanbo/codex_validation/locks/bw46; G=${1:-1}
cd /opt/kda_env && rm -rf g2_p80a && cp -a g2_r5 g2_p80a && cd g2_p80a && rm -rf build
python3 $K/p80a_wb_ablate.py . || exit 1
python3 build_g2.py > /tmp/b80a.log 2>&1 && echo build_ok || { grep -m5 error /tmp/b80a.log; exit 1; }
exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75
hy-smi -d $G --showmemuse 2>&1 | grep -q 'HCU memory use (%): 0' || { echo HBM_BUSY; exit 76; }
export KDA_G2=1 HIP_VISIBLE_DEVICES=$G PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover
cd /tmp
for TH in "16384 12" "65536 12" "16384 48"; do set -- $TH; echo "== T=$1 H=$2"
  python3 $K/op_env_time.py --tree $T --fn wu --T $1 --H $2 --env KDA_G2_WU_NOWB --vals 0,1 --reps 15 2>&1 | grep -E " us |rror"
  [ $2 = 12 ] && python3 $K/op_env_time.py --tree $T --fn dhu_bn --T $1 --H $2 --env KDA_G2_DHU_WBT --vals 0,1 --reps 15 2>&1 | grep -E " us |rror"
done
echo P80A_DONE
