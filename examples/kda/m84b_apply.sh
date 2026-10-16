#!/bin/bash
# apply the P-84b merge to a hip_kda tree (cwd = tree root): wu phase-1 k/g loads issued first, on by default
set -e
K=/public/home/tanbo/kra/examples/kda
python3 $K/p84b_wu_ks.py csrc/g2
sed -i 's/g2::g2_env_long("KDA_G2_WU_KS", 0) != 0;   \/\/ kra P-84b probe/g2::g2_env_long("KDA_G2_WU_KS", 1) != 0;   \/\/ kra P-84b (default on)/' csrc/g2/g2_wu.cuh
grep -q 'KDA_G2_WU_KS", 1' csrc/g2/g2_wu.cuh && echo "P-84b default on"
