#!/bin/bash
# Cross-platform comparison, BW side, DEVICE time via hipprof traces (torch.profiler's
# device_time_total is inflated on DCU at large shapes: G2 8K/48 21.3 ms vs hipprof 15.4 ms).
# Per (shape, arm) N processes of kda_iter.py (5 measured fwd+bwd windows each, spin markers);
# the per-window kernel sum is reduced to a per-process median.  Run on the bw7 HOST:
#   bash xplat_bw_dev.sh GPU "8192:12 8192:48" [N=3] [OUTDIR=/public/home/tanbo/xplat/dev]
GPU=$1; SHAPES=$2; N=${3:-3}; OUT=${4:-/public/home/tanbo/xplat/dev}
mkdir -p "$OUT"
for s in $SHAPES; do
  T=${s%%:*}; H=${s##*:}
  for arm in fla g2; do
    for r in $(seq 1 "$N"); do
      tag="${arm}_${T}_${H}_r${r}"
      docker exec -e HIP_VISIBLE_DEVICES="$GPU" -e KDA_G2=1 -e TRITON_CACHE_DIR=/tmp/triton_mega \
        -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/opt/kda_env:/opt/kda_env/g2_r5:/opt/kda_env/takeover \
        tanbo_mega_k3 bash -lc "rm -f /tmp/xpd_$tag.*; hipprof --hip-trace --output-type 0 -o /tmp/xpd_$tag \
          python3 /public/home/tanbo/kra/examples/kda/kda_iter.py --tree /opt/kda_env/g2_r5 --arm $arm \
          --T $T --H $H --iters 5 > /dev/null 2>&1; cp /tmp/xpd_$tag.json $OUT/ 2>/dev/null"
    done
  done
done
echo "done $SHAPES"
