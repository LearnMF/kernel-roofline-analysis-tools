#!/bin/bash
K=/public/home/tanbo/kra/examples/kda; T=/opt/kda_env/g2_p79c; L=/public/home/tanbo/codex_validation/locks/bw46; G=${1:-1}
cd /opt/kda_env && rm -rf g2_p79c && cp -a g2_r5 g2_p79c && cd g2_p79c && rm -rf build
python3 $K/p78a_s4b_occ.py . && python3 $K/p79_s4b_ot.py . && python3 $K/p79c_s4b_nobar.py . || exit 1
python3 build_g2.py > /tmp/b79c.log 2>&1 && echo build_ok || { grep -m5 error /tmp/b79c.log; exit 1; }
exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75
hy-smi -d $G --showmemuse 2>&1 | grep -q 'HCU memory use (%): 0' || { echo HBM_BUSY; exit 76; }
export KDA_G2=1 HIP_VISIBLE_DEVICES=$G PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover
cd /tmp
for pad in 0 8192; do echo "== 16K/12 PADLDS=$pad"; KDA_G2_S4B_PADLDS=$pad python3 $K/op_env_time.py --tree $T --fn wy_s4_split --T 16384 --H 12 --env KDA_G2_S4B_OT --vals 0,3 --reps 15 2>&1 | grep -E "us |rror"; done
echo P79C_DONE
