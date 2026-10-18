#!/bin/bash
K=/public/home/tanbo/kra/examples/kda; L=/public/home/tanbo/codex_validation/locks/bw46; G=${1:-4}
( exec 9>$L/hcu_$G.lock; flock -w 3600 9 || exit 75; export KDA_G2=1 HIP_VISIBLE_DEVICES=$G; cd /tmp; mkdir -p /tmp/ulp93
  for TH in "8192 12" "16384 48"; do set -- $TH
    for tr in g2_r5 g2_p93q; do
      PYTHONPATH=/opt/kda_env:/opt/kda_env/$tr:/opt/kda_env/takeover python3 $K/bitwise_gate.py dump --tree /opt/kda_env/$tr --T $1 --H $2 --out /tmp/ulp93/${tr}_$1_$2.pt > /dev/null 2>&1
    done
    echo "== $1/$2: B1' vs production"; python3 $K/ulp_diff.py /tmp/ulp93/g2_r5_$1_$2.pt /tmp/ulp93/g2_p93q_$1_$2.pt
  done )
echo ULP93_DONE
