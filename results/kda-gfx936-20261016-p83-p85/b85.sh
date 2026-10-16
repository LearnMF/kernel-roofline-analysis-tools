#!/bin/bash
K=/public/home/tanbo/kra/examples/kda; T=/opt/kda_env/g2_p85; L=/public/home/tanbo/codex_validation/locks/bw46; G=${1:-1}
cd /opt/kda_env && rm -rf g2_p85 && cp -a g2_r5 g2_p85 && cd g2_p85 && rm -rf build
python3 $K/p85_dhu_window.py csrc/g2 || exit 1
python3 build_g2.py > /tmp/b85.log 2>&1 && echo build_ok || { grep -m5 error /tmp/b85.log; exit 1; }
GT=$K/bitwise_gate.py
r=$( exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75; export KDA_G2=1 HIP_VISIBLE_DEVICES=$G CACHE=/tmp/gate_b85
     bash $K/gate_all.sh $T "8192:12 8192:48 32768:48" 2>&1
     KDA_G2_FI=3 PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover python3 $GT dump --tree $T --T 8192 --H 48 --out /tmp/gate_b85/cand_fi3.pt > /tmp/gate_b85/fi3.log 2>&1
     echo "fi3 8192_48: $(python3 $GT cmp /tmp/gate_b85/base_8192_48.pt /tmp/gate_b85/cand_fi3.pt | tail -1)" )
echo "$r" | grep -E "PASS|FAIL" | tr '\n' ' '; echo
[[ "$r" == *"GATE_ALL PASS"* ]] || { echo "GATE FAILED -> no A/B"; echo "$r" | tail -30; exit 1; }
( exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75; export KDA_G2=1 HIP_VISIBLE_DEVICES=$G PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover; cd /tmp
  for TH in "16384 48" "32768 48" "65536 48" "8192 96" "16384 24"; do set -- $TH
    python3 $K/pipe_cfg.py --tree $T --T $1 --H $2 --reps 8 --cfg old:KDA_G2_DHU4_MAX_CTAS=256 --cfg new: 2>&1 | grep -E "cfg|T=|rror"; done )
echo P85_DONE
