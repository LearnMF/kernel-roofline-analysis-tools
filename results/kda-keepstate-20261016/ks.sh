#!/bin/bash
# keep-state re-measure on current production (P-65): wall F+B with KDA_G2_KEEP_STATE 0 vs 1 toggled per rep, plus saved/peak memory
K=/public/home/tanbo/kra/examples/kda; T=/opt/kda_env/g2_r5; L=/public/home/tanbo/codex_validation/locks/bw46; G=$1; shift
exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75
hy-smi -d $G --showmemuse 2>&1 | grep -q 'HCU memory use (%): 0' || { echo HBM_BUSY; exit 76; }
export KDA_G2=1 HIP_VISIBLE_DEVICES=$G PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover
cd /tmp
for TH in "$@"; do set -- $TH; python3 $K/pipe_time.py --tree $T --T $1 --H $2 --env KDA_G2_KEEP_STATE --reps 12 2>&1 | grep "T="; done
