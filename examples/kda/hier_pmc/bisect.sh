#!/usr/bin/env bash
# bisect.sh <hcu> "<pmc line 1>" ["<pmc line 2>"] : which counter set crashes hipprof on the KDA app (8192/12)
set -uo pipefail
dev=$1; shift
L=/public/home/tanbo/codex_validation/locks
exec 8>"$L/bw7/hcu_${dev}.lock"; flock -w 600 8 || { echo LOCK_BUSY; exit 75; }
exec 9>"$L/hcu_${dev}.lock";     flock -w 600 9 || { echo LOCK_BUSY; exit 75; }
TREE=/opt/kda_env/g2_r5
printf 'pmc: %s\n' "$@" > /tmp/hp_bis_$dev.txt
docker cp /tmp/hp_bis_$dev.txt tanbo_mega_k3:/tmp/kra_pmc/bis_$dev.txt >/dev/null
docker exec -e HIP_VISIBLE_DEVICES=$dev -e KDA_G2=1 \
  -e PYTHONPATH=/public/home/tanbo/xplat/bwenv:$TREE:/public/home/tanbo/codex_validation/stage_g2_bw7/takeover \
  tanbo_mega_k3 bash -lc "
export PATH=/opt/dtk/bin:\$PATH; O=/tmp/kra_pmc/bis_d$dev; rm -rf \$O; mkdir -p \$O && cd \$O
HIPPROF_INPUT_FILE=/tmp/kra_pmc/bis_$dev.txt timeout 900 hipprof --pmc --pmc-type 3 -o b python3 /public/home/tanbo/kra/examples/kda/kda_iter.py --tree $TREE --arm g2 --T 8192 --H 12 --iters 1 > b.log 2>&1
echo \"[\$(cat /tmp/kra_pmc/bis_$dev.txt | tr '\n' ';')] csv_lines=\$(wc -l < b.csv 2>/dev/null) signal=\$(grep -c 'signal' b.log)\""
