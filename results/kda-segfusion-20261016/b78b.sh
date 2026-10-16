#!/bin/bash
K=/public/home/tanbo/kra/examples/kda; T=/opt/kda_env/g2_p78b; L=/public/home/tanbo/codex_validation/locks/bw46; G=${1:-7}
cd /opt/kda_env && rm -rf g2_p78b && cp -a g2_r5 g2_p78b && cd g2_p78b && rm -rf build
python3 $K/p78b_prep_occ.py . || exit 1
python3 build_g2.py > /tmp/b78b.log 2>&1 && echo build_ok || { grep -m5 error /tmp/b78a.log; exit 1; }
exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75
hy-smi -d $G --showmemuse 2>&1 | grep -q 'HCU memory use (%): 0' || { echo HBM_BUSY; exit 76; }
export KDA_G2=1 HIP_VISIBLE_DEVICES=$G PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover
cd /tmp
for TH in "16384 12" "16384 48"; do set -- $TH; echo "== T=$1 H=$2"; python3 $K/op_env_time.py --tree $T --fn fwd_prep --T $1 --H $2 --env KDA_G2_PREP_PADLDS --vals 0,21000 --reps 15 2>&1 | grep -E "KDA_G2|rror"; done
echo P78B_DONE
