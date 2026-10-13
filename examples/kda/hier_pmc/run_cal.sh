#!/usr/bin/env bash
set -uo pipefail
L=/public/home/tanbo/codex_validation/locks
exec 8>"$L/bw7/hcu_0.lock"; flock -w 120 8 || { echo LOCK_BUSY; exit 75; }
exec 9>"$L/hcu_0.lock"; flock -w 120 9 || { echo LOCK_BUSY; exit 75; }
docker cp /public/home/tanbo/hier_pmc/pmc_cal.hip tanbo_mega_k3:/tmp/kra_pmc/pmc_cal.hip >/dev/null
docker exec -e HIP_VISIBLE_DEVICES=0 tanbo_mega_k3 bash -lc '
export PATH=/opt/dtk/bin:$PATH; cd /tmp/kra_pmc && rm -f cal.csv cal.log
hipcc -O3 --offload-arch=gfx936 -o pmc_cal pmc_cal.hip 2>&1 | grep -i error
cat > pmc_cal.txt <<EOT
pmc: SQ_INSTS_MMOP SQ_ACTIVE_INST_MMOP SQ_INSTS_LDS SQ_LDS_IDX_ACTIVE SQ_LDS_BANK_CONFLICT SQ_INSTS_VALU SQ_WAVES GRBM_GUI_ACTIVE
EOT
HIPPROF_INPUT_FILE=/tmp/kra_pmc/pmc_cal.txt hipprof --pmc --pmc-type 3 -o cal ./pmc_cal > cal.log 2>&1; grep expected cal.log
python3 - <<EOP
import csv
for r in csv.DictReader(open("cal.csv")):
    t=(int(r["EndNs"])-int(r["BeginNs"]))/1e3
    print(r["KernelName"][:8], "t_us=%.1f"%t, {k:int(float(r[k])) for k in ("SQ_INSTS_MMOP","SQ_ACTIVE_INST_MMOP","SQ_INSTS_LDS","SQ_LDS_IDX_ACTIVE","SQ_LDS_BANK_CONFLICT","SQ_INSTS_VALU","SQ_WAVES","GRBM_GUI_ACTIVE")})
EOP'
