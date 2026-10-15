#!/bin/bash
# P-62 flash-train X1 chain (use_qk_l2norm_in_kernel=False): new FT (r21) vs old FT (r20), dense + varlen, H=12 (kp) and 48
export HIP_VISIBLE_DEVICES=7

K=/public/home/tanbo/kra/examples/kda; HIP=/opt/kda_env/g2_r5; C=/tmp/ft_chk_p62x1; mkdir -p $C
for s in "8192:12:" "8192:12:0,1000,1063,1064,5000,8192" "8192:48:"; do
  IFS=: read -r T H CUS <<<"$s"; tag="${T}_${H}${CUS:+_vl}"; cu=${CUS:+--cu $CUS}
  for v in r21 r22; do
    PYTHONPATH=/opt/kda_env/ft_site_$v:/opt/kda_env:$HIP:/opt/kda_env/takeover python3 $K/ft_dump.py --tree $HIP --T $T --H $H $cu --no-l2norm --out $C/${v}_$tag.pt > $C/${v}_$tag.log 2>&1 || { echo "$tag $v FAILED"; tail -3 $C/${v}_$tag.log; }
  done
  echo "X1 $tag r22 vs r21: $(python3 $K/bitwise_gate.py cmp $C/r21_$tag.pt $C/r22_$tag.pt | tail -1)"
done
