#!/bin/bash
K=/public/home/tanbo/kra/examples/kda; T=/opt/kda_env/g2_p89; L=/public/home/tanbo/codex_validation/locks/bw46; G=${1:-3}
cd /opt/kda_env && rm -rf g2_p89 && cp -a g2_r5 g2_p89 && cd g2_p89 && rm -rf build
grep -q 'KDA_G2_S4_HF", 1' csrc/g2/g2_wy.cuh || { echo "base is not P-88"; exit 1; }
python3 $K/p89_hf_all.py csrc/g2 || exit 1
python3 build_g2.py > /tmp/b89.log 2>&1 && echo build_ok || { grep -m8 error /tmp/b89.log; exit 1; }
GT=$K/bitwise_gate.py
r=$( exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75
     export KDA_G2=1 HIP_VISIBLE_DEVICES=$G CACHE=/tmp/gate_b89 KDA_G2_HF_WU=1 KDA_G2_HF_DAV=1 KDA_G2_HF_TAIL=1 KDA_G2_HF_PREP=1 KDA_G2_HF_GATE=1
     bash $K/gate_all.sh $T "8192:12 8192:48 8192:96" 2>&1
     KDA_G2_FI=3 PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover python3 $GT dump --tree $T --T 8192 --H 12 --out /tmp/gate_b89/cand_fi3.pt > /tmp/gate_b89/fi3.log 2>&1
     echo "fi3 8192_12: $(python3 $GT cmp /tmp/gate_b89/base_8192_12.pt /tmp/gate_b89/cand_fi3.pt | tail -1)" )
echo "$r" | grep -E "PASS|FAIL" | tr '\n' ' '; echo
[[ "$r" == *"GATE_ALL PASS"* ]] || { echo "GATE FAILED -> no A/B"; echo "$r" | tail -30; exit 1; }
( exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75
  export KDA_G2=1 HIP_VISIBLE_DEVICES=$G PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover; cd /tmp
  for TH in "16384 12" "16384 48" "16384 96"; do set -- $TH
    for E in KDA_G2_HF_WU KDA_G2_HF_DAV KDA_G2_HF_TAIL KDA_G2_HF_PREP KDA_G2_HF_GATE; do
      python3 $K/pipe_time.py --tree $T --T $1 --H $2 --env $E --reps 10 2>&1 | grep "T="; done; done )
echo P89_DONE
