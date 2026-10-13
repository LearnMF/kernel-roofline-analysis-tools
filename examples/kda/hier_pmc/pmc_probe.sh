#!/usr/bin/env bash
# test a custom PMC list via HIPPROF_INPUT_FILE inside tanbo_mega_k3 (HCU 0, flock-gated)
set -uo pipefail
L=/public/home/tanbo/codex_validation/locks
exec 8>"$L/bw7/hcu_0.lock"; flock -w 120 8 || { echo LOCK_BUSY; exit 75; }
exec 9>"$L/hcu_0.lock"; flock -w 120 9 || { echo LOCK_BUSY; exit 75; }
docker exec -e HIP_VISIBLE_DEVICES=0 tanbo_mega_k3 bash -lc '
export PATH=/opt/dtk/bin:$PATH; mkdir -p /tmp/kra_pmc && cd /tmp/kra_pmc && rm -rf t1*
cat > pmc_t1.txt <<EOT
pmc: TCC_REQ_sum TCC_READ_sum TCC_WRITE_sum TCC_HIT_sum
pmc: SQ_INSTS_MMOP SQ_INSTS_LDS SQ_INSTS_VALU SQ_ACTIVE_INST_MMOP
EOT
cat > mm.py <<EOT
import torch
a=torch.randn(4096,4096,device="cuda",dtype=torch.bfloat16); b=torch.randn(4096,4096,device="cuda",dtype=torch.bfloat16)
for _ in range(2): c=a@b
torch.cuda.synchronize()
EOT
HIPPROF_INPUT_FILE=/tmp/kra_pmc/pmc_t1.txt timeout 600 hipprof --pmc --pmc-type 3 -o t1 python3 mm.py > t1.log 2>&1; echo rc=$?
ls t1* | head; tail -5 t1.log; f=$(ls t1*.csv 2>/dev/null | head -1); [ -n "$f" ] && head -3 "$f" | cut -c1-700'
