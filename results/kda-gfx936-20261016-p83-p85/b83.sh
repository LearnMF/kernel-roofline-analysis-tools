#!/bin/bash
K=/public/home/tanbo/kra/examples/kda; T=/opt/kda_env/g2_p83; L=/public/home/tanbo/codex_validation/locks/bw46; G=${1:-2}
cd /opt/kda_env && rm -rf g2_p83 && cp -a g2_r5 g2_p83 && cd g2_p83 && rm -rf build
python3 $K/p83_s4a_pf.py csrc/g2 || exit 1
python3 build_g2.py > /tmp/b83.log 2>&1 && echo build_ok || { grep -m8 error /tmp/b83.log; exit 1; }
( mkdir -p /tmp/isa83 && cd /tmp/isa83 && rm -f * && cp $T/hip_kda/_G2*.so g2.so && PATH=/opt/dtk/aillvm/bin:$PATH /opt/dtk/bin/extractkernel -i g2.so > /dev/null 2>&1
  for kn in "p3_wy_s4a_kernelILb0ELb1ELi1E" "p3_wy_s4a_kernelILb0ELb0ELi1E" "p3_wy_s4a_kernelILb0ELb1ELi0E" "p3_wy_s4a_kernelILb1ELb0ELi1E"; do echo "$kn: $(/opt/dtk/aillvm/bin/llvm-readelf --notes g2.so-000-gfx936.hsaco | grep -A40 "$kn" | grep -E "private_segment|vgpr_count|group_segment" | head -3 | tr '\n' ' ')"; done )
GT=$K/bitwise_gate.py
r=$( exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75; export KDA_G2=1 HIP_VISIBLE_DEVICES=$G CACHE=/tmp/gate_b83 KDA_G2_S4A_PF=1
     bash $K/gate_all.sh $T "8192:12 8192:48" 2>&1
     KDA_G2_FI=3 PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover python3 $GT dump --tree $T --T 8192 --H 12 --out /tmp/gate_b83/cand_fi3.pt > /tmp/gate_b83/fi3.log 2>&1
     echo "fi3 8192_12: $(python3 $GT cmp /tmp/gate_b83/base_8192_12.pt /tmp/gate_b83/cand_fi3.pt | tail -1)" )
echo "$r" | grep -E "PASS|FAIL" | tr '\n' ' '; echo
[[ "$r" == *"GATE_ALL PASS"* ]] || { echo "GATE FAILED -> no A/B"; echo "$r" | tail -30; exit 1; }
( exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75; hy-smi -d $G --showmemuse 2>&1 | grep -q 'HCU memory use (%): 0' || { echo HBM_BUSY; exit 76; }
  export KDA_G2=1 HIP_VISIBLE_DEVICES=$G PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover; cd /tmp
  for TH in "16384 12" "65536 12" "16384 48" "8192 96"; do set -- $TH
    python3 $K/op_env_time.py --tree $T --fn wy_s4_split --T $1 --H $2 --env KDA_G2_S4A_PF --vals 0,1 --reps 12 2>&1 | grep -E " us |rror"
    python3 $K/pipe_time.py --tree $T --T $1 --H $2 --env KDA_G2_S4A_PF --reps 10 2>&1 | grep "T="; done )
echo P83_DONE
