#!/bin/bash
# P-90 probe on bw46: tree = production (P-88) + p90; gate vs production; per-kernel A/B in full pipelines (abk46)
K=/public/home/tanbo/kra/examples/kda; X=/public/home/tanbo/xplat; T=/opt/kda_env/g2_p90; L=/public/home/tanbo/codex_validation/locks/bw46; G=${1:-3}
cd /opt/kda_env && rm -rf g2_p90 && cp -a g2_r5 g2_p90 && cd g2_p90 && rm -rf build
python3 $K/p90_prep_beta.py csrc/g2 || exit 1
python3 build_g2.py > /tmp/b90.log 2>&1 && echo build_ok || { grep -m8 error /tmp/b90.log; exit 1; }
r=$( exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75; export KDA_G2=1 HIP_VISIBLE_DEVICES=$G CACHE=/tmp/gate_b90
     bash $K/gate_all.sh $T "8192:12 8192:48 8192:96" 2>&1 )
echo "$r" | grep -E "PASS|FAIL" | tr '\n' ' '; echo
[[ "$r" == *"GATE_ALL PASS"* ]] || { echo "GATE FAILED -> no A/B"; echo "$r" | tail -30; exit 1; }
g=$G; for TH in "16384 12" "16384 48" "16384 96"; do set -- $TH
  bash $X/abk46.sh /opt/kda_env/g2_r5 $T $1 $2 $g $X/b46_p90_$1_$2.txt; done
for TH in 16384_12 16384_48 16384_96; do echo "== $TH"; grep -E "prep_a|TOTAL" $X/b46_p90_$TH.txt; done
echo P90_DONE
