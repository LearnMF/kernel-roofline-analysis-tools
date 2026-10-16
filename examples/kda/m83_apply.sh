#!/bin/bash
# apply the P-85 + P-83 merge to a hip_kda tree (cwd = tree root): dhu 4-wave window 96..127 CTAs, s4a PF on by default
set -e
K=/public/home/tanbo/kra/examples/kda
python3 $K/p85_dhu_window.py csrc/g2
python3 $K/p83_s4a_pf.py csrc/g2
sed -i 's/g2::g2_env_long("KDA_G2_S4A_PF", 0) != 0;   \/\/ kra P-83 probe/g2::g2_env_long("KDA_G2_S4A_PF", 1) != 0;   \/\/ kra P-83 (default on)/' csrc/g2/g2_wy.cuh
grep -q 'KDA_G2_S4A_PF", 1' csrc/g2/g2_wy.cuh && echo "P-83 default on"
