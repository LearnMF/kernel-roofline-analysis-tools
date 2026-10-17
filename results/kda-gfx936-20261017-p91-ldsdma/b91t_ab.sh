#!/bin/bash
K=/public/home/tanbo/kra/examples/kda; X=/public/home/tanbo/xplat; L=/public/home/tanbo/codex_validation/locks; G=${1:-5}
export PP_PRE=/public/home/tanbo/xplat/bwenv PP_POST=/public/home/tanbo/codex_validation/stage_g2_bw7/takeover
GT=$K/bitwise_gate.py
r=$( exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75; export KDA_G2=1 HIP_VISIBLE_DEVICES=$G CACHE=/tmp/gate_b91t BASE=/public/home/tanbo/g2_opt
     bash $K/gate_all.sh /public/home/tanbo/g2_p91t "8192:12 8192:48" 2>&1 )
echo "$r" | grep -E "PASS|FAIL" | tr '\n' ' '; echo
[[ "$r" == *"GATE_ALL PASS"* ]] || { echo "GATE FAILED -> no A/B"; echo "$r" | tail -30; exit 1; }
( exec 9>$L/hcu_$G.lock; flock -w 900 9 || exit 75
  export KDA_G2=1 KDA_SKIP_FLA_PRUNE=1 HIP_VISIBLE_DEVICES=$G JITTER=1 KDA_G2_S4B_PADLDS=12288; cd /tmp
  for TH in "16384 12" "16384 48"; do set -- $TH
    # A = register staging at 1 CTA/CU (pad), B = tri DMA (40 KB static LDS -> 1 CTA/CU, no pad)
    bash $K/ab_kernels.sh /public/home/tanbo/g2_p91pad /public/home/tanbo/g2_p91t $1 $2 5 6 /tmp/abk91t_$G > $X/b52_p91t_$1_$2.raw 2>&1 || true
    echo "== 1 CTA/CU: register staging (A) vs tri DMA (B) $1/$2"; grep -E "s4b|TOTAL common" $X/b52_p91t_$1_$2.raw; done )
echo P91T_DONE
