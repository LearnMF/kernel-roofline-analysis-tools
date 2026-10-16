#!/bin/bash
# P-75 screening on bw46: build, scratch check, per-SS bitwise + dhu replay timing (16K/12, 64K/12), SQTT per SS
K=/public/home/tanbo/kra/examples/kda; T=/opt/kda_env/g2_p75; X=/public/home/tanbo/xplat; L=/public/home/tanbo/codex_validation/locks/bw46
G=${1:-7}
cd /opt/kda_env && rm -rf g2_p75 && cp -a g2_r5 g2_p75 && cd g2_p75 && rm -rf build
python3 $K/p75_dhu_spread.py . || exit 1
python3 build_g2.py > /tmp/b75.log 2>&1 && echo build_ok || { grep -m5 error /tmp/b75.log; exit 1; }
( mkdir -p /tmp/isa75 && cd /tmp/isa75 && rm -f * && cp $T/hip_kda/_G2*.so g2.so && PATH=/opt/dtk/aillvm/bin:$PATH /opt/dtk/bin/extractkernel -i g2.so > /dev/null 2>&1
  for ss in 1 2 3; do echo "SS=$ss $(/opt/dtk/aillvm/bin/llvm-readelf --notes g2.so-000-gfx936.hsaco | grep -A40 "g2_dhu_v8b_kernelILi1ELb1ELi${ss}E" | grep -E "private_segment|vgpr_count" | head -2 | tr '\n' ' ')"; done )
exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75
hy-smi -d $G --showmemuse 2>&1 | grep -q 'HCU memory use (%): 0' || { echo HBM_BUSY; exit 76; }
export KDA_G2=1 HIP_VISIBLE_DEVICES=$G PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover
cd /tmp
for TH in "16384 12" "65536 12"; do set -- $TH; echo "== T=$1 H=$2"; python3 $K/dhu_env_time.py --tree $T --T $1 --H $2 --vals 0,1,2,3 --reps 20 2>&1 | grep -E "KDA_G2_DHU_SS|rror"; done
echo P75_DONE
