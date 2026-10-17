#!/bin/bash
K=/public/home/tanbo/kra/examples/kda; T=/opt/kda_env/g2_r5; L=/public/home/tanbo/codex_validation/locks/bw46; G=${1:-3}
exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75
export KDA_G2=1 HIP_VISIBLE_DEVICES=$G PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover; cd /tmp
for TH in "32768 48" "65536 48" "16384 96" "32768 24" "8192 24" "4096 96"; do set -- $TH
  hy-smi -d $G --showmemuse 2>&1 | grep -o "use (%): [0-9]*" | head -1
  python3 $K/pipe_cfg.py --tree $T --T $1 --H $2 --reps 8 --cfg base: --cfg h:KDA_G2_SAVE_H_MAX_H=96 \
     --cfg kph:KDA_G2_KPERM_MAX_H=96,KDA_G2_SAVE_H_MAX_H=96 2>&1 | grep -E "cfg|T=|rror"; done
echo B87M_DONE
