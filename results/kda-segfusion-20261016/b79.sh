#!/bin/bash
K=/public/home/tanbo/kra/examples/kda; T=/opt/kda_env/g2_p79; L=/public/home/tanbo/codex_validation/locks/bw46; G=${1:-7}
cd /opt/kda_env && rm -rf g2_p79 && cp -a g2_r5 g2_p79 && cd g2_p79 && rm -rf build
python3 $K/p78a_s4b_occ.py . && python3 $K/p79_s4b_ot.py . || exit 1
python3 build_g2.py > /tmp/b79.log 2>&1 && echo build_ok || { grep -m5 error /tmp/b79.log; exit 1; }
( mkdir -p /tmp/isa79 && cd /tmp/isa79 && rm -f * && cp $T/hip_kda/_G2*.so g2.so && PATH=/opt/dtk/aillvm/bin:$PATH /opt/dtk/bin/extractkernel -i g2.so > /dev/null 2>&1
  for ot in 1 2; do echo "OT=$ot $(/opt/dtk/aillvm/bin/llvm-readelf --notes g2.so-000-gfx936.hsaco | grep -A40 "p3_wy_s4b_kernelILb0ELi${ot}E" | grep -E "private_segment|vgpr_count" | head -2 | tr '\n' ' ')"; done )
exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75
hy-smi -d $G --showmemuse 2>&1 | grep -q 'HCU memory use (%): 0' || { echo HBM_BUSY; exit 76; }
export KDA_G2=1 HIP_VISIBLE_DEVICES=$G PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover
cd /tmp
for TH in "16384 12" "16384 48"; do set -- $TH
  for pad in 0 8192; do echo "== T=$1 H=$2 PADLDS=$pad"; KDA_G2_S4B_PADLDS=$pad python3 $K/op_env_time.py --tree $T --fn wy_s4_split --T $1 --H $2 --env KDA_G2_S4B_OT --vals 0,1,2 --reps 15 2>&1 | grep -E "KDA_G2|rror"; done; done
echo P79_DONE
