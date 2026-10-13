#!/usr/bin/env bash
# collect.sh <hcu 0..7> <T> <H> [TREE]  -- run on bw7 HOST.  Collects, for one KDA shape:
#   hy-smi snapshot (before+after), 5x hip-trace (iters 6, KRA_JITTER_SEED=1..5),
#   hipprof --pmc-read / --pmc-write presets, and two custom PMC passes
#   (L2: TCC_REQ/READ/WRITE/HIT;  SQ: SQ_INSTS_MMOP SQ_ACTIVE_INST_MMOP SQ_LDS_IDX_ACTIVE).
# Raw output -> /public/home/tanbo/kra_cards/<T>_<H>/ (NFS, fetch to Mac afterwards).
# Resumable: existing non-empty outputs are skipped.  Aborts (rc 75) if the lock is
# busy >10min, (rc 80) if the sanity run crashes, (rc 81) if a PMC pass yields no csv.
set -uo pipefail
dev=$1 T=$2 H=$3 TREE=${4:-/opt/kda_env/g2_card_p52} OUT=${5:-kra_cards}
L=/public/home/tanbo/codex_validation/locks
exec 8>"$L/bw7/hcu_${dev}.lock"; flock -w 600 8 || { echo LOCK_BUSY; exit 75; }
exec 9>"$L/hcu_${dev}.lock";     flock -w 600 9 || { echo LOCK_BUSY; exit 75; }
docker exec -e HIP_VISIBLE_DEVICES=$dev -e KDA_G2=1 -e KDA_SKIP_FLA_PRUNE=1 \
  -e PYTHONPATH=/opt/kda_env:/public/home/tanbo/xplat/bwenv:$TREE:/public/home/tanbo/codex_validation/stage_g2_bw7/takeover \
  tanbo_mega_k3 bash -lc "
export PATH=/opt/dtk/bin:\$PATH
O=/public/home/tanbo/$OUT/${T}_${H}; mkdir -p \$O && cd \$O
hy-smi --showuse --showmeminfo vram > hysmi_before.txt 2>&1 || true
echo 'pmc: TCC_REQ_sum TCC_READ_sum TCC_WRITE_sum TCC_HIT_sum' > pmc_l2.txt
echo 'pmc: SQ_INSTS_MMOP SQ_ACTIVE_INST_MMOP SQ_LDS_IDX_ACTIVE' > pmc_sq.txt
echo 'fingerprint=35b7bd3a8862 tree=$TREE' > provenance.txt
K='python3 /public/home/tanbo/hier_pmc/kda_run.py --tree $TREE --arm g2 --T $T --H $H'
nonempty() { [ -s \"\$1\" ]; }
# sanity (skip if trace_1 already captured)
if ! nonempty \$O/trace_1.json; then
  timeout 900 \$K --iters 1 > sanity.log 2>&1 || { echo SANITY_FAIL; tail -5 sanity.log; exit 80; }
fi
for i in 1 2 3 4 5; do
  ok=0
  if nonempty \$O/trace_\$i.json && grep -q KDA_ITER_DONE \$O/trace_\$i.log; then ok=1; fi
  [ \$ok -eq 1 ] || { rm -f \$O/trace_\$i.db \$O/trace_\$i.json; KRA_JITTER_SEED=\$i timeout 1500 hipprof --hip-trace --output-type 0 -o \$O/trace_\$i \$K --iters 6 > trace_\$i.log 2>&1; echo trace_\$i rc=\$?; grep -q KDA_ITER_DONE \$O/trace_\$i.log || echo trace_\$i INCOMPLETE; }
done
{ nonempty \$O/pmc_read.csv  && grep -q KDA_ITER_DONE \$O/pmc_read.log; }  || { rm -f \$O/pmc_read.db \$O/pmc_read.csv;  timeout 1500 hipprof --pmc-read  --pmc-type 3 -o \$O/pmc_read  \$K --iters 1 > pmc_read.log  2>&1; echo pmc_read  rc=\$?; }
{ nonempty \$O/pmc_write.csv && grep -q KDA_ITER_DONE \$O/pmc_write.log; } || { rm -f \$O/pmc_write.db \$O/pmc_write.csv; timeout 1500 hipprof --pmc-write --pmc-type 3 -o \$O/pmc_write \$K --iters 1 > pmc_write.log 2>&1; echo pmc_write rc=\$?; }
{ nonempty \$O/pmc_l2.csv    && grep -q KDA_ITER_DONE \$O/pmc_l2.log; }    || { rm -f \$O/pmc_l2.db \$O/pmc_l2.csv;    HIPPROF_INPUT_FILE=\$O/pmc_l2.txt timeout 1500 hipprof --pmc --pmc-type 3 -o \$O/pmc_l2 \$K --iters 1 > pmc_l2.log 2>&1; echo pmc_l2 rc=\$?; }
{ nonempty \$O/pmc_sq.csv    && grep -q KDA_ITER_DONE \$O/pmc_sq.log; }    || { rm -f \$O/pmc_sq.db \$O/pmc_sq.csv;    HIPPROF_INPUT_FILE=\$O/pmc_sq.txt timeout 1500 hipprof --pmc --pmc-type 3 -o \$O/pmc_sq \$K --iters 1 > pmc_sq.log 2>&1; echo pmc_sq rc=\$?; }
nonempty \$O/optrace.json || { PYTHONPATH=/public/home/tanbo/kra:\$PYTHONPATH timeout 900 \$K --iters 1 --optrace \$O/optrace.json > optrace.log 2>&1; echo optrace rc=\$?; }
hy-smi --showuse --showmeminfo vram > hysmi_after.txt 2>&1 || true
ls -l \$O | tail -12"
