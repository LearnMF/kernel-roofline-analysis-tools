#!/bin/bash
# merge P-82 into bw7 (bw47) /opt/kda_env/g2_r5 (container tanbo_mega_k3)
set -e
cd /opt/kda_env
test ! -e g2_r5_bak_p82
cp -a g2_r5 g2_r5_bak_p82
cd g2_r5 && python3 /public/home/tanbo/kra/examples/kda/p82_save_h.py .
python3 build_g2.py > /tmp/b82m.log 2>&1 && echo build_ok
echo "fp $(sha256sum csrc/g2/g2_fwd_v8.cuh csrc/g2/g2_fwd_prep.cuh csrc/g2/g2_wu.cuh csrc/g2/g2_bindings.cpp hip_kda/ops/k3_g2.py hip_kda/ops/g2_ops.py | sha256sum | cut -c1-12)"
set +e
L=/public/home/tanbo/codex_validation/locks/bw7; exec 9>$L/hcu_4.lock; flock -w 900 9 || { echo LOCK_BUSY; exit 75; }
export KDA_G2=1 HIP_VISIBLE_DEVICES=4 BASE=/opt/kda_env/g2_r5_bak_p82 CACHE=/tmp/gate_b82m
bash /public/home/tanbo/kra/examples/kda/gate_all.sh /opt/kda_env/g2_r5 "8192:12 8192:48" 2>&1 | grep -E "GATE|FAIL" | tr '\n' ' '; echo
