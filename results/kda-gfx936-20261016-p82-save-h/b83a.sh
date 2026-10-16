#!/bin/bash
# extend option d to H > 20: does the kp layout + MODE 5 kp/BP instance pay at TP4/TP2/TP1?  (production P-82 tree)
K=/public/home/tanbo/kra/examples/kda; T=/opt/kda_env/g2_r5; L=/public/home/tanbo/codex_validation/locks/bw46; G=${1:-1}
exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75
hy-smi -d $G --showmemuse 2>&1 | grep -q 'HCU memory use (%): 0' || { echo HBM_BUSY; exit 76; }
export KDA_G2=1 HIP_VISIBLE_DEVICES=$G PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover
cd /tmp
for TH in "16384 24" "16384 48" "32768 48" "8192 96"; do set -- $TH; echo "== T=$1 H=$2"
  python3 $K/pipe_cfg.py --tree $T --T $1 --H $2 --reps 10 --cfg base: --cfg kp:KDA_G2_KPERM_MAX_H=96 \
     --cfg kph:KDA_G2_KPERM_MAX_H=96,KDA_G2_SAVE_H_MAX_H=96 --cfg h:KDA_G2_SAVE_H_MAX_H=96 2>&1 | grep -E "cfg|T=|rror"
done
echo P83A_DONE
