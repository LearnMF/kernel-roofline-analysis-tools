#!/bin/bash
# Cross-platform tracker, BW side, flash-train route: device time (hipprof kernel sum per fwd+bwd window,
# kda_iter.py, placement jitter R20) of hip_kda P-88 (g2 arm, hip_kda.fla.chunk_kda) and flash-train feat/kda
# (ftc arm, flash_train.torch.chunk_kda) interleaved process by process with the same jitter seeds.
#   bash round_ftgh.sh OUTDIR [N=5] [GPUS="4 5 6 7"]
OUT=$1; N=${2:-5}; GPUS=${3:-"4 5 6 7"}
SHAPES=${SHAPES:-"8192:12 8192:24 8192:48 8192:96 16384:12 16384:24 16384:48 16384:96 32768:12 32768:24 32768:48 32768:96 65536:12 65536:24 65536:48 65536:96"}
KRA=/public/home/tanbo/kra; TREE=/public/home/tanbo/xplat/ftgh_tree; FT=${FT_SITE:-/public/home/tanbo/xplat/ft_site_ghf}
mkdir -p "$OUT/dev_g2" "$OUT/dev_ftc"
export KDA_G2=1 TRITON_CACHE_DIR=/tmp/triton_round_ft PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$FT:/public/home/tanbo/xplat/bwenv:/public/home/tanbo/g2_opt:/public/home/tanbo/codex_validation/stage_g2_bw7/takeover
i=0; declare -A Q
ORDER=$(for s in $SHAPES; do echo "$(( ${s%%:*} * ${s##*:} )) $s"; done | sort -rn | awk "{print \$2}")
for s in $ORDER; do g=$(echo $GPUS | cut -d" " -f$((i % $(echo $GPUS | wc -w) + 1))); Q[$g]="${Q[$g]} $s"; i=$((i + 1)); done
devrun() {  # arm T H r gpu
  local tag="g2_${2}_${3}_r${4}"; local tmp=/tmp/rft_${1}_$tag; rm -f $tmp.*
  KRA_JITTER_SEED=$4 HIP_VISIBLE_DEVICES=$5 hipprof --hip-trace --output-type 0 -o $tmp python3 $KRA/examples/kda/kda_iter.py \
    --tree $TREE --arm $1 --T $2 --H $3 --iters 5 > /dev/null 2>&1
  cp $tmp.json "$OUT/dev_$1/xpd_${tag}.json" 2>/dev/null || echo "FAIL $1 $2 $3 r$4"; rm -f $tmp.*
}
for g in "${!Q[@]}"; do
  ( for s in ${Q[$g]}; do T=${s%%:*}; H=${s##*:}
      for r in $(seq 1 "$N"); do
        if (( r % 2 == 0 )); then devrun ftc $T $H $r $g; devrun g2 $T $H $r $g; else devrun g2 $T $H $r $g; devrun ftc $T $H $r $g; fi
      done; echo "done $s gpu$g $(date +%T)"
    done ) &
done
wait
cd $KRA && for a in g2 ftc; do PYTHONPATH=$KRA python3 examples/kda/xplat_bw_dev_parse.py "$OUT/dev_$a" > "$OUT/dev_$a.json"; done
echo ROUND_FTGH_DONE
