#!/bin/bash
# P-92 at equal occupancy: register path forced to 1 CTA/CU (PADLDS=12288, p91pad) vs P-92 (60 KB static, no knob)
K=/public/home/tanbo/kra/examples/kda; X=/public/home/tanbo/xplat; L=/public/home/tanbo/codex_validation/locks; G=${1:-5}
export PP_PRE=/public/home/tanbo/xplat/bwenv PP_POST=/public/home/tanbo/codex_validation/stage_g2_bw7/takeover
cd /public/home/tanbo && rm -rf g2_p92n && cp -a g2_opt g2_p92n && cd g2_p92n && rm -rf build
python3 $K/p92_s4b_async.py csrc/g2 --no-pad || exit 1
grep -c KDA_G2_S4B_PADLDS csrc/g2/g2_wy.cuh
python3 build_g2.py > /tmp/b92n.log 2>&1 && echo build_ok || { grep -m12 error /tmp/b92n.log; exit 1; }
( exec 9>$L/hcu_$G.lock; flock -w 3600 9 || exit 75
  export KDA_G2=1 KDA_SKIP_FLA_PRUNE=1 HIP_VISIBLE_DEVICES=$G JITTER=1 KDA_G2_S4B_PADLDS=12288; cd /tmp
  for TH in "16384 12" "16384 48"; do set -- $TH
    bash $K/ab_kernels.sh /public/home/tanbo/g2_p91pad /public/home/tanbo/g2_p92n $1 $2 5 6 /tmp/abk92n_$G > $X/b52_p92n_$1_$2.txt 2>&1
    echo "== 1 CTA/CU both: register path (A) vs P-92 (B) $1/$2"; grep -E "s4b|TOTAL common" $X/b52_p92n_$1_$2.txt; done )
echo P92N_DONE
