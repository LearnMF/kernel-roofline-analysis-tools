#!/bin/bash
K=/public/home/tanbo/kra/examples/kda; T=/opt/kda_env/g2_p81a; L=/public/home/tanbo/codex_validation/locks/bw46; G=${1:-1}
cd /opt/kda_env && rm -rf g2_p81a && cp -a g2_r5 g2_p81a && cd g2_p81a && rm -rf build
python3 $K/p81a_mode5.py . || exit 1
python3 build_g2.py > /tmp/b81a.log 2>&1 && echo build_ok || { grep -m5 error /tmp/b81a.log; exit 1; }
exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75
hy-smi -d $G --showmemuse 2>&1 | grep -q 'HCU memory use (%): 0' || { echo HBM_BUSY; exit 76; }
export KDA_G2=1 HIP_VISIBLE_DEVICES=$G PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover
cd /tmp
for TH in "16384 12" "65536 12" "16384 48"; do set -- $TH; echo "== T=$1 H=$2"
  python3 $K/op_variant_time.py --tree $T --fn fwd_h_o --T $1 --H $2 --variant base: --variant bn:emit_bn=1 --check bn:2=fwd_h_bn:0 --reps 12 2>&1 | grep -E "check|fwd_h_o |rror"
  python3 $K/op_variant_time.py --tree $T --fn wu --T $1 --H $2 --variant base: --variant nop1:emit_p1=0 --reps 12 2>&1 | grep -E "wu |rror"
done
echo P81A_DONE
