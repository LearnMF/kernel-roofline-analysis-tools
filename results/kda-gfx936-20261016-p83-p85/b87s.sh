#!/bin/bash
# clean re-measure on bw52 (production g2_opt): saved-h without kp / with kp beyond today's policy
K=/public/home/tanbo/kra/examples/kda; T=/public/home/tanbo/g2_opt; L=/public/home/tanbo/codex_validation/locks; G=$1; shift
exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75
export KDA_G2=1 HIP_VISIBLE_DEVICES=$G PYTHONPATH=/public/home/tanbo/xplat/bwenv:$T:/public/home/tanbo/codex_validation/stage_g2_bw7/takeover; cd /tmp
for TH in "$@"; do set -- $TH
  echo "hcu$G $(hy-smi -d $G --showmemuse 2>&1 | grep -o 'use (%): [0-9]*' | head -1)"
  python3 $K/pipe_cfg.py --tree $T --T $1 --H $2 --reps 8 --cfg base: --cfg h:KDA_G2_SAVE_H_MAX_H=96 \
     --cfg kph:KDA_G2_KPERM_MAX_H=96,KDA_G2_SAVE_H_MAX_H=96 2>&1 | grep -E "cfg|T=|rror"
  echo "hcu$G after $(hy-smi -d $G --showmemuse 2>&1 | grep -o 'use (%): [0-9]*' | head -1)"; done
echo B87S_DONE
