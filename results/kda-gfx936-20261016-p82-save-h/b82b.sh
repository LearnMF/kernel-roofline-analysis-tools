#!/bin/bash
K=/public/home/tanbo/kra/examples/kda; T=/opt/kda_env/g2_p82b; L=/public/home/tanbo/codex_validation/locks/bw46; G=${1:-1}
cd /opt/kda_env && rm -rf g2_p82b && cp -a g2_r5 g2_p82b && cd g2_p82b
python3 $K/p82b_save_h_policy.py . || exit 1
GT=$K/bitwise_gate.py
r=$( exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75; export KDA_G2=1 HIP_VISIBLE_DEVICES=$G CACHE=/tmp/gate_b82b
     bash $K/gate_all.sh $T "8192:12 8192:48 32768:48" 2>&1 )
echo "$r" | grep -E "PASS|FAIL" | tr '\n' ' '; echo
[[ "$r" == *"GATE_ALL PASS"* ]] || { echo "GATE FAILED -> no A/B"; echo "$r" | tail -30; exit 1; }
( exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75; export KDA_G2=1 HIP_VISIBLE_DEVICES=$G PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover; cd /tmp
  for TH in "16384 24" "32768 48" "65536 48" "16384 48"; do set -- $TH
    python3 $K/pipe_cfg.py --tree $T --T $1 --H $2 --reps 8 --cfg off:KDA_G2_SAVE_H=0 --cfg on: 2>&1 | grep -E "cfg|T=|rror"; done
  for TH in "16384 12" "16384 24"; do set -- $TH
    python3 $K/pipe_cfg.py --tree $T --T $1 --H $2 --reps 8 --cfg ks_mode2:KDA_G2_KEEP_STATE=1,KDA_G2_SAVE_H=0 --cfg ks_h:KDA_G2_KEEP_STATE=1 --cfg k0:KDA_G2_SAVE_H=0 2>&1 | grep -E "cfg|T=|rror"; done )
echo P82B_DONE
