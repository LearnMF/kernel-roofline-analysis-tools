#!/bin/bash
# P-54 incremental collection (task F5): production tree g2_card_p54.
# Only what the delta needs: 5 jittered hip-traces + pmc-read + pmc-write.
#   bash collect_p54.sh <hcu> <T> <H>
set -euo pipefail
dev=$1 T=$2 H=$3 TREE=/opt/kda_env/g2_card_p54
L=/public/home/tanbo/codex_validation/locks
exec 8>"$L/bw7/hcu_${dev}.lock"; flock -w 600 8 || { echo LOCK_BUSY; exit 75; }
exec 9>"$L/hcu_${dev}.lock";     flock -w 600 9 || { echo LOCK_BUSY; exit 75; }
docker exec -e HIP_VISIBLE_DEVICES=$dev -e KDA_G2=1 -e KDA_SKIP_FLA_PRUNE=1 \
  -e PYTHONPATH=/opt/kda_env:/public/home/tanbo/xplat/bwenv:$TREE:/public/home/tanbo/codex_validation/stage_g2_bw7/takeover \
  tanbo_mega_k3 bash -lc "
export PATH=/opt/dtk/bin:\$PATH
O=/public/home/tanbo/kra_cards/p54_${T}_${H}; mkdir -p \$O && cd \$O
echo 'fingerprint=90a91fdd5fdd tree=$TREE (g2_r5 @P-53+P-54a)' > provenance.txt
K='python3 /public/home/tanbo/hier_pmc/kda_run.py --tree $TREE --arm g2 --T $T --H $H'
nonempty() { [ -s \"\$1\" ]; }
if ! nonempty \$O/trace_1.json; then
  timeout 900 \$K --iters 1 > sanity.log 2>&1 || { echo SANITY_FAIL; tail -5 sanity.log; exit 80; }
fi
for i in 1 2 3 4 5; do
  ok=0
  if nonempty \$O/trace_\$i.json && grep -q KDA_ITER_DONE \$O/trace_\$i.log; then ok=1; fi
  [ \$ok -eq 1 ] || { rm -f \$O/trace_\$i.db \$O/trace_\$i.json; KRA_JITTER_SEED=\$i timeout 1500 hipprof --hip-trace --output-type 0 -o \$O/trace_\$i \$K --iters 6 > trace_\$i.log 2>&1; echo trace_\$i rc=\$?; grep -q KDA_ITER_DONE \$O/trace_\$i.log || echo trace_\$i INCOMPLETE; }
done
{ nonempty \$O/pmc_read.csv  && grep -q KDA_ITER_DONE \$O/pmc_read.log; }  || { rm -f \$O/pmc_read.db \$O/pmc_read.csv;  timeout 1500 hipprof --pmc-read  --pmc-type 3 -o \$O/pmc_read  \$K --iters 1 > pmc_read.log  2>&1; echo pmc_read  rc=\$?; }
{ nonempty \$O/pmc_write.csv && grep -q KDA_ITER_DONE \$O/pmc_write.log; } || { rm -f \$O/pmc_write.db \$O/pmc_write.csv; timeout 1500 hipprof --pmc-write --pmc-type 3 -o \$O/pmc_write \$K --iters 1 > pmc_write.log  2>&1; echo pmc_write rc=\$?; }
echo DONE_${T}_${H}
"
