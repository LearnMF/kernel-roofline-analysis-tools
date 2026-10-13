#!/usr/bin/env bash
# run_kda.sh <hcu 0..3> <T> <H> : hierarchical-roofline PMC for production G2 (g2_r5).
# Two hipprof invocations: pass S (SQ/LDS/MMAC, 7 counters) then pass M (TCC L2 + HBM EA, 9 counters).
# Total counters per invocation must stay <= ~10 or rocprof Construct() aborts the app.
set -uo pipefail
dev=$1 T=$2 H=$3
L=/public/home/tanbo/codex_validation/locks
exec 8>"$L/bw7/hcu_${dev}.lock"; flock -w 600 8 || { echo LOCK_BUSY; exit 75; }
exec 9>"$L/hcu_${dev}.lock";     flock -w 600 9 || { echo LOCK_BUSY; exit 75; }
TREE=/opt/kda_env/g2_r5
docker exec -e HIP_VISIBLE_DEVICES=$dev -e KDA_G2=1 -e KDA_SKIP_FLA_PRUNE=1 \
  -e PYTHONPATH=/public/home/tanbo/xplat/bwenv:$TREE:/public/home/tanbo/codex_validation/stage_g2_bw7/takeover \
  tanbo_mega_k3 bash -lc "
export PATH=/opt/dtk/bin:\$PATH; O=/tmp/kra_pmc/kda_${T}_${H}; rm -rf \$O; mkdir -p \$O && cd \$O
echo 'pmc: SQ_INSTS_MMOP SQ_ACTIVE_INST_MMOP SQ_LDS_IDX_ACTIVE SQ_LDS_BANK_CONFLICT SQ_INSTS_LDS SQ_INSTS_VALU GRBM_GUI_ACTIVE' > pmc_s.txt
echo 'pmc: TCC_REQ_sum TCC_READ_sum TCC_WRITE_sum TCC_HIT_sum TCC_EA_RDREQ_sum TCC_EA_RDREQ_32B_sum TCC_EA_WRREQ_sum TCC_EA_WRREQ_64B_sum TCC_REQ_sum' > pmc_m.txt
for pass in s m; do
  HIPPROF_INPUT_FILE=\$O/pmc_\$pass.txt timeout 1500 hipprof --pmc --pmc-type 3 -o hr_\$pass python3 /public/home/tanbo/hier_pmc/kda_run.py --tree $TREE --arm g2 --T $T --H $H --iters 1 > hr_\$pass.log 2>&1; echo \$pass rc=\$?
done
wc -l hr_*.csv; grep -c Construct hr_s.log hr_m.log; tail -2 hr_s.log hr_m.log"
