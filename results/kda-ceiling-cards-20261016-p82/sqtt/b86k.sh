#!/bin/bash
# non-kp MODE 5 question on the current production: kp layouts + saved h (BP MODE 5) vs today's policy at H > 24
K=/public/home/tanbo/kra/examples/kda; T=/opt/kda_env/g2_r5; L=/public/home/tanbo/codex_validation/locks/bw46; G=${1:-2}
exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75
export KDA_G2=1 HIP_VISIBLE_DEVICES=$G PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover; cd /tmp
for TH in "8192 48" "16384 48" "16384 32" "8192 96" "16384 96"; do set -- $TH
  python3 $K/pipe_cfg.py --tree $T --T $1 --H $2 --reps 8 --cfg base: --cfg kph:KDA_G2_KPERM_MAX_H=96,KDA_G2_SAVE_H_MAX_H=96 \
     --cfg h:KDA_G2_SAVE_H_MAX_H=96 2>&1 | grep -E "cfg|T=|rror"; done
echo B86K_DONE
