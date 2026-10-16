#!/bin/bash
# SQTT diagnosis batch on production (P-82b): s4a TP8/TP2, wu TP8, dhu v8b4 TP2; plus dhu variant timing at H=48
X=/public/home/tanbo/xplat; K=/public/home/tanbo/kra/examples/kda; T=/opt/kda_env/g2_r5; G=${1:-1}
bash $X/cap_k3.sh p3_wy_s4a_kernel s4a_16k12 $G $T 16384 12 > /dev/null 2>&1; echo s4a_16k12 done
bash $X/cap_k3.sh p3_wy_s4a_kernel s4a_8k48 $G $T 8192 48 > /dev/null 2>&1; echo s4a_8k48 done
bash $X/cap_k3.sh g2_wu_kernel wu_16k12 $G $T 16384 12 > /dev/null 2>&1; echo wu_16k12 done
bash $X/cap_k3.sh g2_dhu_v8b4_kernel dhu4_16k48 $G $T 16384 48 > /dev/null 2>&1; echo dhu4_16k48 done
L=/public/home/tanbo/codex_validation/locks/bw46; exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75
export KDA_G2=1 HIP_VISIBLE_DEVICES=$G PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover; cd /tmp
for TH in "16384 48" "32768 48"; do set -- $TH; echo "== dhu T=$1 H=$2"
  python3 $K/op_env_time.py --tree $T --fn dhu_bn --T $1 --H $2 --env KDA_G2_R1A_DHU --vals 1,0 --reps 10 2>&1 | grep -E " us |rror"
  python3 $K/op_env_time.py --tree $T --fn dhu_bn --T $1 --H $2 --env KDA_G2_DHU_THROTTLE_MIN_CTAS --vals 192,99999 --reps 10 2>&1 | grep -E " us |rror"
  KDA_G2_R1A_DHU=0 python3 $K/op_env_time.py --tree $T --fn dhu_bn --T $1 --H $2 --env KDA_G2_DHU_THROTTLE_MIN_CTAS --vals 192,99999 --reps 10 2>&1 | grep -E " us |rror" | sed "s/^/[8-wave] /"
done
echo DIAG_DONE
