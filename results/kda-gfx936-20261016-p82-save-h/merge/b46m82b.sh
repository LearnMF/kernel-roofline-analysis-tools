#!/bin/bash
# bring bw46's probe base /opt/kda_env/g2_r5 to production P-82b (container tanbo_kda_b46)
set -e
cd /opt/kda_env
test ! -e g2_r5_bak_p82b
cp -a g2_r5 g2_r5_bak_p82b
cd g2_r5 && python3 /public/home/tanbo/kra/examples/kda/p82b_save_h_policy.py . && echo no_build_python_only
echo "fp $(sha256sum csrc/g2/g2_fwd_v8.cuh csrc/g2/g2_fwd_prep.cuh csrc/g2/g2_wu.cuh csrc/g2/g2_bindings.cpp hip_kda/ops/k3_g2.py hip_kda/ops/g2_ops.py | sha256sum | cut -c1-12)"
