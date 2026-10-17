#!/bin/bash
K=/public/home/tanbo/kra/examples/kda; T=/opt/kda_env/g2_p83b; L=/public/home/tanbo/codex_validation/locks/bw46; G=${1:-2}
cd /opt/kda_env && rm -rf g2_p83b && cp -a g2_r5 g2_p83b && cd g2_p83b && rm -rf build
python3 $K/p83b_s4a_afs.py csrc/g2 || exit 1
python3 build_g2.py > /tmp/b83b.log 2>&1 && echo build_ok || { grep -m8 error /tmp/b83b.log; exit 1; }
( mkdir -p /tmp/isa83b && cd /tmp/isa83b && rm -f * && cp $T/hip_kda/_G2*.so g2.so && PATH=/opt/dtk/aillvm/bin:$PATH /opt/dtk/bin/extractkernel -i g2.so > /dev/null 2>&1
  echo "PF2: $(/opt/dtk/aillvm/bin/llvm-readelf --notes g2.so-000-gfx936.hsaco | grep -B2 -A40 "p3_wy_s4a_kernelILb0ELb1ELi2E" | grep -E "private_segment|vgpr_count" | head -2 | tr '\n' ' ')" )
GT=$K/bitwise_gate.py
r=$( exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75; export KDA_G2=1 HIP_VISIBLE_DEVICES=$G CACHE=/tmp/gate_b83b KDA_G2_S4A_AFSR=1
     bash $K/gate_all.sh $T "8192:12" 2>&1 )
echo "$r" | grep -E "PASS|FAIL" | tr '\n' ' '; echo
[[ "$r" == *"GATE_ALL PASS"* ]] || { echo "GATE FAILED -> no A/B"; echo "$r" | tail -30; exit 1; }
( exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75; export KDA_G2=1 HIP_VISIBLE_DEVICES=$G PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover; cd /tmp
  for TH in "16384 12" "65536 12"; do set -- $TH
    python3 $K/op_env_time.py --tree $T --fn wy_s4_split --T $1 --H $2 --env KDA_G2_S4A_AFSR --vals 0,1 --reps 12 2>&1 | grep -E " us |rror"
    python3 $K/pipe_time.py --tree $T --T $1 --H $2 --env KDA_G2_S4A_AFSR --reps 10 2>&1 | grep "T="; done )
echo P83B_DONE
