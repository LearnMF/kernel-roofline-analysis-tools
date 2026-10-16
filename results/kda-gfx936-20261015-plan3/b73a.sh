#!/bin/bash
K=/public/home/tanbo/kra/examples/kda; X=/public/home/tanbo/xplat
for v in all g; do
  ( cd /opt/kda_env && rm -rf g2_p73a_$v && cp -a g2_r5 g2_p73a_$v && cd g2_p73a_$v && rm -rf build && python3 $K/p73a_s4b_ablate.py . $v > /dev/null && python3 build_g2.py > /tmp/b73a_$v.log 2>&1 && echo "$v build_ok" ) || echo "$v build FAILED"
done
bash $X/abk46.sh /opt/kda_env/g2_r5 /opt/kda_env/g2_p73a_all 16384 12 1 $X/b46_p73a_all_16384_12.txt &
bash $X/abk46.sh /opt/kda_env/g2_r5 /opt/kda_env/g2_p73a_g 16384 12 2 $X/b46_p73a_g_16384_12.txt &
bash $X/abk46.sh /opt/kda_env/g2_r5 /opt/kda_env/g2_p73a_all 16384 48 3 $X/b46_p73a_all_16384_48.txt &
wait; echo P73A_DONE
