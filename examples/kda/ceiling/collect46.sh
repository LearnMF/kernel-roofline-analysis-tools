#!/usr/bin/env bash
# collect46.sh <hcu> <T> <H> [TREE] [OUT] -- bw46 variant of ceiling/collect.sh, run INSIDE tanbo_kda_b46.
# 5x hip-trace (iters 6, KRA_JITTER_SEED=1..5), pmc-read / pmc-write presets, custom L2 + SQ passes, optrace.
set -uo pipefail
dev=$1 T=$2 H=$3 TREE=${4:-/opt/kda_env/g2_r5} OUT=${5:-kra_cards_p82}
L=/public/home/tanbo/codex_validation/locks/bw46
exec 9>"$L/hcu_${dev}.lock"; flock -w 900 9 || { echo LOCK_BUSY; exit 75; }
hy-smi -d $dev --showmemuse 2>&1 | grep -q 'HCU memory use (%): 0' || { echo HBM_BUSY; exit 76; }
export HIP_VISIBLE_DEVICES=$dev KDA_G2=1 KDA_SKIP_FLA_PRUNE=1 PYTHONPATH=/opt/kda_env:$TREE:/opt/kda_env/takeover PATH=/opt/dtk/bin:$PATH
O=/public/home/tanbo/$OUT/${T}_${H}; mkdir -p $O && cd $O
hy-smi --showuse --showmeminfo vram > hysmi_before.txt 2>&1 || true
echo 'pmc: TCC_REQ_sum TCC_READ_sum TCC_WRITE_sum TCC_HIT_sum' > pmc_l2.txt
echo 'pmc: SQ_INSTS_MMOP SQ_ACTIVE_INST_MMOP SQ_LDS_IDX_ACTIVE' > pmc_sq.txt
echo "tree=$TREE (P-82b production) host=bw46 hcu=$dev" > provenance.txt
K="python3 /public/home/tanbo/hier_pmc/kda_run.py --tree $TREE --arm g2 --T $T --H $H"
nonempty() { [ -s "$1" ]; }
nonempty $O/trace_1.json || { timeout 900 $K --iters 1 > sanity.log 2>&1 || { echo SANITY_FAIL; tail -5 sanity.log; exit 80; }; }
for i in 1 2 3 4 5; do
  if nonempty $O/trace_$i.json && grep -q KDA_ITER_DONE $O/trace_$i.log; then continue; fi
  rm -f $O/trace_$i.db $O/trace_$i.json; KRA_JITTER_SEED=$i timeout 1500 hipprof --hip-trace --output-type 0 -o $O/trace_$i $K --iters 6 > trace_$i.log 2>&1; echo trace_$i rc=$?
done
for pm in read write; do
  { nonempty $O/pmc_$pm.csv && grep -q KDA_ITER_DONE $O/pmc_$pm.log; } || { rm -f $O/pmc_$pm.db $O/pmc_$pm.csv; timeout 1500 hipprof --pmc-$pm --pmc-type 3 -o $O/pmc_$pm $K --iters 1 > pmc_$pm.log 2>&1; echo pmc_$pm rc=$?; }
done
for pm in l2 sq; do
  { nonempty $O/pmc_$pm.csv && grep -q KDA_ITER_DONE $O/pmc_$pm.log; } || { rm -f $O/pmc_$pm.db $O/pmc_$pm.csv; HIPPROF_INPUT_FILE=$O/pmc_$pm.txt timeout 1500 hipprof --pmc --pmc-type 3 -o $O/pmc_$pm $K --iters 1 > pmc_$pm.log 2>&1; echo pmc_$pm rc=$?; }
done
nonempty $O/optrace.json || { PYTHONPATH=/public/home/tanbo/kra:$PYTHONPATH timeout 900 $K --iters 1 --optrace $O/optrace.json > optrace.log 2>&1; echo optrace rc=$?; }
hy-smi --showuse --showmeminfo vram > hysmi_after.txt 2>&1 || true
ls $O | tr '\n' ' '; echo; echo COLLECT_DONE_${T}_${H}
