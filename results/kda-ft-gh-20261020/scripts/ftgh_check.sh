#!/bin/bash
# Bitwise gate (inside tanbo_mega_bw52): new FT (feat/kda: P-77b + P-82 ports) default / recompute / keep-state
# vs old FT (ft_site_a1e = a1e0e35) and vs hip_kda P-88 (g2_opt 6bbdc76). Args: GPU
set -u
export HIP_VISIBLE_DEVICES=$1 KDA_G2=1 PYTHONDONTWRITEBYTECODE=1 TRITON_CACHE_DIR=/tmp/triton_ftgh
K=/public/home/tanbo/kra/examples/kda; HIP=/public/home/tanbo/g2_opt
PP=/public/home/tanbo/xplat/bwenv:$HIP:/public/home/tanbo/codex_validation/stage_g2_bw7/takeover
NEW=${FT_SITE:-/public/home/tanbo/xplat/ft_site_gh}; OLD=/public/home/tanbo/xplat/ft_site_a1e; C=${CHK:-/tmp/ftgh_chk}; mkdir -p $C
RC="FLASH_TRAIN_KDA_SAVE_H=0 FLASH_TRAIN_KDA_SAVE_QKN=0 FLASH_TRAIN_KDA_KEEP_STATE=0"
KS="FLASH_TRAIN_KDA_SAVE_H=0 FLASH_TRAIN_KDA_KEEP_STATE=1"
fail=0
for spec in "8192:12:" "8192:48:" "8192:96:" "32768:12:" "32768:48:" "8192:12:0,1000,1063,1064,5000,8192" "8192:48:0,63,4097,8192" "8192:96:0,4096,8192"; do
  IFS=: read -r T H CUS <<<"$spec"
  tag="${T}_${H}${CUS:+_vl$(echo $CUS | md5sum | cut -c1-4)}"; cu=${CUS:+--cu $CUS}
  [ -e $C/hip_$tag.pt ] || PYTHONPATH=$PP python3 $K/bitwise_gate.py dump --tree $HIP --T $T --H $H $cu --out $C/hip_$tag.pt > $C/hip_$tag.log 2>&1 || { echo "$tag hip FAILED"; fail=1; continue; }
  [ -e $C/old_$tag.pt ] || PYTHONPATH=$OLD:$PP python3 $K/ft_dump.py --tree $HIP --T $T --H $H $cu --out $C/old_$tag.pt > /dev/null 2>&1
  for v in def rc ks; do
    case $v in def) E="";; rc) E=$RC;; ks) E=$KS;; esac
    env $E PYTHONPATH=$NEW:$PP python3 $K/ft_dump.py --tree $HIP --T $T --H $H $cu --out $C/${v}_$tag.pt > $C/${v}_$tag.log 2>&1 || { echo "$tag $v FAILED"; tail -3 $C/${v}_$tag.log; fail=1; continue; }
    r1=$(python3 $K/bitwise_gate.py cmp $C/hip_$tag.pt $C/${v}_$tag.pt | tail -1)
    r2=$(python3 $K/bitwise_gate.py cmp $C/old_$tag.pt $C/${v}_$tag.pt | tail -1)
    echo "$tag new[$v] vs hip_kda: $r1 | vs old FT: $r2"
  done
done
echo "FTGH_CHECK_DONE fail=$fail"
