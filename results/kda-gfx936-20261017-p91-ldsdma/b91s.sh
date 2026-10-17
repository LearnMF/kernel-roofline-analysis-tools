#!/bin/bash
# P-91 prototype on bw52 (tanbo_mega_bw52): trees g2_p91pad (production + occupancy knob) and g2_p91 (+ DMA staging)
K=/public/home/tanbo/kra/examples/kda; X=/public/home/tanbo/xplat; L=/public/home/tanbo/codex_validation/locks; G=${1:-3}
export PP_PRE=/public/home/tanbo/xplat/bwenv PP_POST=/public/home/tanbo/codex_validation/stage_g2_bw7/takeover
cd /public/home/tanbo
for v in p91pad p91; do
  rm -rf g2_$v && cp -a g2_opt g2_$v && cd g2_$v && rm -rf build
  grep -q 'KDA_G2_S4_HF", 1' csrc/g2/g2_wy.cuh || { echo "base is not P-88"; exit 1; }
  if [ $v = p91pad ]; then python3 $K/p91_s4b_dma.py csrc/g2 --pad-only || exit 1; else python3 $K/p91_s4b_dma.py csrc/g2 || exit 1; fi
  python3 build_g2.py > /tmp/b91_$v.log 2>&1 && echo "build_ok $v" || { grep -m8 error /tmp/b91_$v.log; exit 1; }
  cd /public/home/tanbo
done
( mkdir -p /tmp/isa91 && cd /tmp/isa91 && rm -f * && cp /public/home/tanbo/g2_p91/hip_kda/_G2*.so g2.so && PATH=/opt/dtk/aillvm/bin:$PATH /opt/dtk/bin/extractkernel -i g2.so > /dev/null 2>&1
  /opt/dtk/aillvm/bin/llvm-readelf --notes g2.so-000-gfx936.hsaco | grep -B3 -A60 "p3_wy_s4b_kernelILb0E" | grep -E "group_segment_fixed_size|vgpr_count|private_segment" | head -3 | tr '\n' ' '; echo
  echo "lds-dma instructions: $(/opt/dtk/aillvm/bin/llvm-objdump -d g2.so-000-gfx936.hsaco 2>/dev/null | grep -c 'buffer_load_dwordx4.*lds')" )
GT=$K/bitwise_gate.py
r=$( exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75; export KDA_G2=1 HIP_VISIBLE_DEVICES=$G CACHE=/tmp/gate_b91 BASE=/public/home/tanbo/g2_opt
     bash $K/gate_all.sh /public/home/tanbo/g2_p91 "8192:12 8192:48 8192:96" 2>&1
     export KDA_G2_S4B_PADLDS=12288
     PYTHONPATH=$PP_PRE:/public/home/tanbo/g2_p91:$PP_POST python3 $GT dump --tree /public/home/tanbo/g2_p91 --T 8192 --H 48 --out /tmp/gate_b91/cand_pad.pt > /tmp/gate_b91/pad.log 2>&1
     echo "pad1cta 8192_48: $(python3 $GT cmp /tmp/gate_b91/base_8192_48.pt /tmp/gate_b91/cand_pad.pt | tail -1)" )
echo "$r" | grep -E "PASS|FAIL" | tr '\n' ' '; echo
[[ "$r" == *"GATE_ALL PASS"* ]] || { echo "GATE FAILED -> no A/B"; echo "$r" | tail -30; exit 1; }
( exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75
  export KDA_G2=1 KDA_SKIP_FLA_PRUNE=1 HIP_VISIBLE_DEVICES=$G JITTER=1; cd /tmp
  for PAD in 0 12288; do for TH in "16384 12" "16384 48"; do set -- $TH
    KDA_G2_S4B_PADLDS=$PAD bash $K/ab_kernels.sh /public/home/tanbo/g2_p91pad /public/home/tanbo/g2_p91 $1 $2 5 6 /tmp/abk91_$G > $X/b52_p91_pad${PAD}_$1_$2.txt 2>&1
    echo "== pad=$PAD $1/$2"; grep -E "s4b|TOTAL common" $X/b52_p91_pad${PAD}_$1_$2.txt; done; done )
echo P91_DONE
