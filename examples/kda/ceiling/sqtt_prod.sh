#!/bin/bash
# SQTT capture of the PRODUCTION driver (kda_run.py / kda_iter.py), not bn_sqtt.
# Run on the bw7 HOST.  Uses the static xprof copied into the prod container.
#
#   bash sqtt_prod.sh <hcu 0..7> <T> <H> [TREE=/opt/kda_env/g2_card_p52]
#
# Produces /public/home/tanbo/kra_cards/sqtt_prod_<T>_<H>/  (xprof .perf + log).
set -euo pipefail
dev=$1 T=$2 H=$3 TREE=${4:-/opt/kda_env/g2_card_p52}
L=/public/home/tanbo/codex_validation/locks
exec 8>"$L/bw7/hcu_${dev}.lock"; flock -w 600 8 || { echo LOCK_BUSY; exit 75; }
exec 9>"$L/hcu_${dev}.lock";     flock -w 600 9 || { echo LOCK_BUSY; exit 75; }
docker exec -e HIP_VISIBLE_DEVICES=$dev -e KDA_G2=1 -e KDA_SKIP_FLA_PRUNE=1 \
  -e PYTHONPATH=/opt/kda_env:/public/home/tanbo/xplat/bwenv:$TREE:/public/home/tanbo/codex_validation/stage_g2_bw7/takeover \
  tanbo_mega_k3 bash -lc "
export PATH=/opt/dtk/bin:\$PATH
export LD_LIBRARY_PATH=/public/home/tanbo/kra_cards/xprof_bin/rocm633/lib:\$LD_LIBRARY_PATH
XP=/public/home/tanbo/kra_cards/xprof_bin/rocm633/bin/xprof
O=/public/home/tanbo/kra_cards/sqtt_prod_${T}_${H}; mkdir -p \$O && cd \$O
K='python3 /public/home/tanbo/hier_pmc/kda_run.py --tree $TREE --arm g2 --T $T --H $H --warmup 1 --iters 1'
# pass 1: enumerate dispatches that match production kernel names
timeout 900 \$XP --kernels g2_,p3_ --output-dir \$O/list \$K --iters 1 > \$O/list.log 2>&1 || true
grep -o 'dispatch kernel \[[0-9]*\]' \$O/list.log | grep -o '[0-9]*' | sort -n > \$O/disp.txt
cat \$O/disp.txt | wc -l
# pass 2: SQTT on the second-half dispatches (iter rep; warmup rep comes first)
N=\$(wc -l < \$O/disp.txt)
LO=\$(awk -v n=\$N 'NR == int(n/2) + 1 {print; exit}' \$O/disp.txt)
HI=\$(tail -1 \$O/disp.txt); HI=\$((HI + 1))
echo \"dispatches rep2: \$LO:\$HI\"
timeout 1800 \$XP --enable-sqtt --kernels g2_,p3_ --dispatches \$LO:\$HI --output-dir \$O/sqtt \$K --iters 1 > \$O/sqtt.log 2>&1; echo sqtt rc=\$?
ls -la \$O/sqtt
"
