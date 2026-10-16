#!/bin/bash
# P-82 (option d) on bw46: build, wu VGPR/scratch check, full bitwise gates vs production, in-process wall A/B
# (KDA_G2_SAVE_H 0 vs 1), per-kernel A/B vs production
K=/public/home/tanbo/kra/examples/kda; T=/opt/kda_env/g2_p82; X=/public/home/tanbo/xplat; L=/public/home/tanbo/codex_validation/locks/bw46
G=${1:-1}
cd /opt/kda_env && rm -rf g2_p82 && cp -a g2_r5 g2_p82 && cd g2_p82 && rm -rf build
python3 $K/p82_save_h.py . || exit 1
python3 build_g2.py > /tmp/b82.log 2>&1 && echo build_ok || { grep -m8 -E "error" /tmp/b82.log; exit 1; }
( mkdir -p /tmp/isa82 && cd /tmp/isa82 && rm -f * && cp $T/hip_kda/_G2*.so g2.so && PATH=/opt/dtk/aillvm/bin:$PATH /opt/dtk/bin/extractkernel -i g2.so > /dev/null 2>&1
  for kn in g2_wu_kernelILb0E g2_wu_kernelILb1E "g2_fwd_h_v8o4_kernelILb0ELi5E"; do echo "$kn: $(/opt/dtk/aillvm/bin/llvm-readelf --notes g2.so-000-gfx936.hsaco | grep -A40 "$kn" | grep -E "private_segment|vgpr_count|group_segment" | head -3 | tr '\n' ' ')"; done )
GT=$K/bitwise_gate.py
r=$( exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75; export KDA_G2=1 HIP_VISIBLE_DEVICES=$G CACHE=/tmp/gate_b82
     bash $K/gate_all.sh $T "8192:12 8192:48" 2>&1
     KDA_G2_FI=3 PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover python3 $GT dump --tree $T --T 8192 --H 12 --out /tmp/gate_b82/cand_fi3.pt > /tmp/gate_b82/fi3.log 2>&1
     echo "fi3 8192_12: $(python3 $GT cmp /tmp/gate_b82/base_8192_12.pt /tmp/gate_b82/cand_fi3.pt | tail -1)" )
echo "$r" | grep -E "PASS|FAIL" | tr '\n' ' '; echo
[[ "$r" == *"GATE_ALL PASS"* ]] || { echo "GATE FAILED -> no A/B"; echo "$r" | tail -40; exit 1; }
( exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75; export KDA_G2=1 HIP_VISIBLE_DEVICES=$G PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover; cd /tmp
  for TH in "16384 12" "65536 12" "16384 48" "8192 96"; do set -- $TH; python3 $K/pipe_time.py --tree $T --T $1 --H $2 --env KDA_G2_SAVE_H --reps 12 2>&1 | grep "T="; done )
g=1; for TH in "16384 12" "65536 12" "16384 48"; do set -- $TH
  bash $X/abk46.sh /opt/kda_env/g2_r5 $T $1 $2 $g $X/b46_p82_$1_$2.txt & g=$((g+1)); done
wait; echo AB_DONE
