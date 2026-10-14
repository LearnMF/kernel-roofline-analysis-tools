#!/bin/bash
# P-61e flash-train X1 chain (use_qk_l2norm_in_kernel=False): new FT (r21) vs old FT (r20), dense + varlen, H=12 (kp) and 48
export HIP_VISIBLE_DEVICES=7
exec 9>/public/home/tanbo/codex_validation/locks/bw7/hcu_7.lock; flock -w 600 9 || { echo LOCK_BUSY; exit 75; }
K=/public/home/tanbo/kra/examples/kda; HIP=/opt/kda_env/g2_r5; C=/tmp/ft_chk_p61x1; mkdir -p $C
for s in "8192:12:" "8192:12:0,1000,1063,1064,5000,8192" "8192:48:"; do
  IFS=: read -r T H CUS <<<"$s"; tag="${T}_${H}${CUS:+_vl}"; cu=${CUS:+--cu $CUS}
  for v in r20 r21; do
    PYTHONPATH=/opt/kda_env/ft_site_$v:/opt/kda_env:$HIP:/opt/kda_env/takeover python3 $K/ft_dump.py --tree $HIP --T $T --H $H $cu --no-l2norm --out $C/${v}_$tag.pt > $C/${v}_$tag.log 2>&1 || { echo "$tag $v FAILED"; tail -3 $C/${v}_$tag.log; }
  done
  echo "X1 $tag r21 vs r20: $(python3 $K/bitwise_gate.py cmp $C/r20_$tag.pt $C/r21_$tag.pt | tail -1)"
done
