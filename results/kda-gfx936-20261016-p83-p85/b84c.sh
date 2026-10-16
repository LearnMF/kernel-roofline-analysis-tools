#!/bin/bash
# P-84b per-kernel A/B in full F+B pipelines: A = P-82b production backup, B = P-84b with KS on by default
X=/public/home/tanbo/xplat
cd /opt/kda_env && rm -rf g2_p84b_on && cp -a g2_p84b g2_p84b_on && cd g2_p84b_on
sed -i 's/g2::g2_env_long("KDA_G2_WU_KS", 0) != 0;/g2::g2_env_long("KDA_G2_WU_KS", 1) != 0;/' csrc/g2/g2_wu.cuh
grep -q 'KDA_G2_WU_KS", 1' csrc/g2/g2_wu.cuh || exit 1
python3 build_g2.py > /tmp/b84c.log 2>&1 && echo build_ok || exit 1
for i in 1 2 3 4 5 6; do [ -d /opt/kda_env/g2_r5_bak_p83 ] && break; sleep 30; done
g=2; for TH in "16384 12" "16384 48" "32768 48"; do set -- $TH
  bash $X/abk46.sh /opt/kda_env/g2_r5_bak_p83 /opt/kda_env/g2_p84b_on $1 $2 $g $X/b46_p84b_$1_$2.txt & g=$((g+1)); done
wait; echo AB_DONE
