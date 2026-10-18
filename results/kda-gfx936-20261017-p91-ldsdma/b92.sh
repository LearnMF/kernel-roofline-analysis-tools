#!/bin/bash
# P-92 full-async s4b prototype on bw52: build, ISA check, gate, A/B vs the register path at 2 and 1 CTA/CU
K=/public/home/tanbo/kra/examples/kda; X=/public/home/tanbo/xplat; L=/public/home/tanbo/codex_validation/locks; G=${1:-3}
export PP_PRE=/public/home/tanbo/xplat/bwenv PP_POST=/public/home/tanbo/codex_validation/stage_g2_bw7/takeover
cd /public/home/tanbo && rm -rf g2_p92 && cp -a g2_opt g2_p92 && cd g2_p92 && rm -rf build
python3 $K/p92_s4b_async.py csrc/g2 || exit 1
python3 build_g2.py > /tmp/b92.log 2>&1 && echo build_ok || { grep -m12 -B2 -A3 error /tmp/b92.log | head -40; exit 1; }
( mkdir -p /tmp/isa92 && cd /tmp/isa92 && rm -f * && cp /public/home/tanbo/g2_p92/hip_kda/_G2*.so g2.so && PATH=/opt/dtk/aillvm/bin:$PATH /opt/dtk/bin/extractkernel -i g2.so > /dev/null 2>&1
  /opt/dtk/aillvm/bin/llvm-readelf --notes g2.so-000-gfx936.hsaco > notes.txt
  n=$(grep -n ".name:.*p3_wy_s4b_kernelILb0E" notes.txt | head -1 | cut -d: -f1); sed -n "$((n-25)),$((n+25))p" notes.txt | grep -E "group_segment|vgpr_count|private_segment" | head -3 | tr '\n' ' '; echo
  /opt/dtk/aillvm/bin/llvm-objdump -d g2.so-000-gfx936.hsaco > all.dis 2>/dev/null
  s=$(grep -n "<_ZN2g216p3_wy_s4b_kernelILb0EEEvNS_11WyS4BParamsE>:" all.dis | cut -d: -f1); sed -n "$s,$((s+9000))p" all.dis | awk 'NR>1 && /^[0-9a-f]+ <_Z/{exit} {print}' > s4b.dis
  echo "lds-dma: $(grep -c 'offen lds' s4b.dis)  global_load: $(grep -c global_load s4b.dis)  vmcnt waits: $(grep -c 's_waitcnt vmcnt' s4b.dis)"
  grep -n "s_waitcnt vmcnt\|offen lds\|s_barrier\|global_load\|global_store\|s_cbranch_scc\|s_cbranch_vcc" s4b.dis | awk '{print $1, $2, $3}' | sed -n 1,200p > waits.txt; wc -l waits.txt )
GT=$K/bitwise_gate.py
r=$( exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75; export KDA_G2=1 HIP_VISIBLE_DEVICES=$G CACHE=/tmp/gate_b92 BASE=/public/home/tanbo/g2_opt
     bash $K/gate_all.sh /public/home/tanbo/g2_p92 "8192:12 8192:48 8192:96" 2>&1 )
echo "$r" | grep -E "PASS|FAIL" | tr '\n' ' '; echo
[[ "$r" == *"GATE_ALL PASS"* ]] || { echo "GATE FAILED -> no A/B"; echo "$r" | grep -v BITWISE | tail -40; exit 1; }
( exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75
  export KDA_G2=1 KDA_SKIP_FLA_PRUNE=1 HIP_VISIBLE_DEVICES=$G JITTER=1; cd /tmp
  for PAD in 0 8192; do for TH in "16384 12" "16384 48"; do set -- $TH
    KDA_G2_S4B_PADLDS=$PAD bash $K/ab_kernels.sh /public/home/tanbo/g2_p91pad /public/home/tanbo/g2_p92 $1 $2 5 6 /tmp/abk92_$G > $X/b52_p92_pad${PAD}_$1_$2.txt 2>&1
    echo "== pad=$PAD (A = register path $( [ $PAD = 0 ] && echo '2 CTA/CU' || echo '1 CTA/CU'), B = P-92 1 CTA/CU) $1/$2"; grep -E "s4b|TOTAL common" $X/b52_p92_pad${PAD}_$1_$2.txt; done; done )
echo P92_DONE
