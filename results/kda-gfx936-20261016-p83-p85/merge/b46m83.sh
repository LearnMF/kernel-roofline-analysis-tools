#!/bin/bash
# bring bw46's probe base /opt/kda_env/g2_r5 to production P-83+P-85 (container tanbo_kda_b46)
set -e
cd /opt/kda_env
test ! -e g2_r5_bak_p83
cp -a g2_r5 g2_r5_bak_p83
cd g2_r5 && bash /public/home/tanbo/kra/examples/kda/m83_apply.sh && python3 build_g2.py > /tmp/b46m83.log 2>&1 && echo build_ok
echo "fp $(sha256sum csrc/g2/g2_fwd_v8.cuh csrc/g2/g2_fwd_prep.cuh csrc/g2/g2_wu.cuh csrc/g2/g2_bindings.cpp hip_kda/ops/k3_g2.py hip_kda/ops/g2_ops.py | sha256sum | cut -c1-12)"
