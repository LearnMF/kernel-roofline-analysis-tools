#!/bin/bash
# P-76 on bw46: build, scratch check, bitwise gates per UV (vs production g2_r5), recompute replay timing per UV
K=/public/home/tanbo/kra/examples/kda; T=/opt/kda_env/g2_p76; L=/public/home/tanbo/codex_validation/locks/bw46; G=${1:-7}
cd /opt/kda_env && rm -rf g2_p76 && cp -a g2_r5 g2_p76 && cd g2_p76 && rm -rf build
python3 $K/p76_rec_u_prefetch.py . || exit 1
python3 build_g2.py > /tmp/b76.log 2>&1 && echo build_ok || { grep -m5 error /tmp/b76.log; exit 1; }
( mkdir -p /tmp/isa76 && cd /tmp/isa76 && rm -f * && cp $T/hip_kda/_G2*.so g2.so && PATH=/opt/dtk/aillvm/bin:$PATH /opt/dtk/bin/extractkernel -i g2.so > /dev/null 2>&1
  for uv in 1 2; do echo "UV=$uv $(/opt/dtk/aillvm/bin/llvm-readelf --notes g2.so-000-gfx936.hsaco | grep -A40 "g2_fwd_h_v8o4_kernelILb0ELi4ELb1ELb1ELb0ELb1ELb0ELb1ELi${uv}E" | grep -E "private_segment|vgpr_count" | head -2 | tr '\n' ' ')"; done )
exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75
hy-smi -d $G --showmemuse 2>&1 | grep -q 'HCU memory use (%): 0' || { echo HBM_BUSY; exit 76; }
export KDA_G2=1 HIP_VISIBLE_DEVICES=$G
for uv in 1 2; do
  r=$( export KDA_G2_REC_UV=$uv CACHE=/tmp/gate_b76_$uv; bash $K/gate_all.sh $T "8192:12 8192:48" 2>&1 )
  echo "UV=$uv gate: $(echo "$r" | grep -E "GATE_ALL" | tail -1)"
done
export PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover
cd /tmp
for TH in "16384 12" "65536 12"; do set -- $TH; echo "== T=$1 H=$2"; python3 $K/op_env_time.py --tree $T --fn fwd_h_bn --T $1 --H $2 --env KDA_G2_REC_UV --vals 0,1,2 --reps 20 2>&1 | grep -E "KDA_G2_REC_UV|rror"; done
echo P76_DONE
