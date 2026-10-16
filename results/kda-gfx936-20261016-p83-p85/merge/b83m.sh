#!/bin/bash
# merge P-83+P-85 into bw7 (bw47) /opt/kda_env/g2_r5 (container tanbo_mega_k3)
set -e
cd /opt/kda_env
test ! -e g2_r5_bak_p83
cp -a g2_r5 g2_r5_bak_p83
cd g2_r5 && bash /public/home/tanbo/kra/examples/kda/m83_apply.sh
python3 build_g2.py > /tmp/b83m.log 2>&1 && echo build_ok
echo "fp $(sha256sum csrc/g2/g2_fwd_v8.cuh csrc/g2/g2_fwd_prep.cuh csrc/g2/g2_wu.cuh csrc/g2/g2_bindings.cpp hip_kda/ops/k3_g2.py hip_kda/ops/g2_ops.py | sha256sum | cut -c1-12)"
set +e
L=/public/home/tanbo/codex_validation/locks/bw7; exec 9>$L/hcu_5.lock; flock -w 900 9 || { echo LOCK_BUSY; exit 75; }
export KDA_G2=1 HIP_VISIBLE_DEVICES=5 BASE=/opt/kda_env/g2_r5_bak_p83 CACHE=/tmp/gate_b83m
bash /public/home/tanbo/kra/examples/kda/gate_all.sh /opt/kda_env/g2_r5 "8192:12 8192:48 32768:48" 2>&1 | grep -E "GATE|FAIL" | tr '\n' ' '; echo
