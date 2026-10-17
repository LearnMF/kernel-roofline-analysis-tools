#!/bin/bash
# apply the P-83b + P-84c merge (cwd = the dir holding g2_wy.cuh / g2_wu.cuh): s4a AFS stages A from pav, wu phase 3
# without Kn uses the carried exp2f(g) with q first -- both on by default
set -e
K=/public/home/tanbo/kra/examples/kda
python3 $K/p83b_s4a_afs.py .
sed -i 's/if (pf \&\& g2::g2_env_long("KDA_G2_S4A_AFSR", 0))/if (pf \&\& g2::g2_env_long("KDA_G2_S4A_AFSR", 1))/' g2_wy.cuh
grep -q 'KDA_G2_S4A_AFSR", 1' g2_wy.cuh && echo "P-83b default on"
python3 $K/p84c_wu_qs.py .
sed -i 's/g2::g2_env_long("KDA_G2_WU_QS", 0) != 0;   \/\/ kra P-84c probe/g2::g2_env_long("KDA_G2_WU_QS", 1) != 0;   \/\/ kra P-84c (default on)/' g2_wu.cuh
grep -q 'KDA_G2_WU_QS", 1' g2_wu.cuh && echo "P-84c default on"
