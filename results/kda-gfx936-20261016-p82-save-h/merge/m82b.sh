#!/bin/bash
# merge P-82b into bw52 production g2_opt (container part; git commit on the host)
set -e
cd /public/home/tanbo
test ! -e g2_opt_bak_p82b
cp -a g2_opt g2_opt_bak_p82b
cd g2_opt && python3 /public/home/tanbo/kra/examples/kda/p82b_save_h_policy.py .
echo no_build_python_only
set +e
L=/public/home/tanbo/codex_validation/locks; exec 9>$L/hcu_0.lock; flock -w 900 9 || { echo LOCK_BUSY; exit 75; }
export KDA_G2=1 HIP_VISIBLE_DEVICES=0 PP_PRE=/public/home/tanbo/xplat/bwenv PP_POST=/public/home/tanbo/codex_validation/stage_g2_bw7/takeover BASE=/public/home/tanbo/g2_opt_bak_p82b CACHE=/tmp/gate_m82
bash /public/home/tanbo/kra/examples/kda/gate_all.sh /public/home/tanbo/g2_opt "8192:12 8192:48 32768:48" 2>&1 | grep -E "GATE|FAIL" | tr '\n' ' '; echo
G=/public/home/tanbo/kra/examples/kda/bitwise_gate.py
KDA_G2_FI=3 PYTHONPATH=$PP_PRE:/public/home/tanbo/g2_opt:$PP_POST python3 $G dump --tree /public/home/tanbo/g2_opt --T 8192 --H 12 --out $CACHE/cand_fi3.pt > $CACHE/fi3.log 2>&1 || echo "fi3 FAILED"
echo "fi3 8192_12: $(python3 $G cmp $CACHE/base_8192_12.pt $CACHE/cand_fi3.pt | tail -1)"
sha256sum csrc/g2/g2_fwd_v8.cuh csrc/g2/g2_fwd_prep.cuh csrc/g2/g2_wu.cuh csrc/g2/g2_bindings.cpp hip_kda/ops/k3_g2.py hip_kda/ops/g2_ops.py | sha256sum | cut -c1-12
