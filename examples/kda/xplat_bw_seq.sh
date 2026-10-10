#!/bin/bash
# Clean BW re-run: everything SEQUENTIAL on one GPU (parallel shape jobs contended for the host
# CPU: wall-time spreads up to 55% and device-time spreads up to 11x in the first pass).
# Per shape: N wall-time bench processes (fla,g2 + checkpoint form), then N hipprof device-time
# processes per arm.  Run on the bw7 HOST:  bash xplat_bw_seq.sh GPU [N=3] [DIR=/public/home/tanbo/xplat/seq]
GPU=$1; N=${2:-3}; DIR=${3:-/public/home/tanbo/xplat/seq}
SHAPES="8192:12 8192:24 8192:48 8192:96 16384:12 32768:12"
mkdir -p "$DIR/dev"
for s in $SHAPES; do
  bash /public/home/tanbo/kra/examples/kda/xplat_bw.sh "$GPU" "$s" "$N" "$DIR/bw.jsonl"
  bash /public/home/tanbo/kra/examples/kda/xplat_bw_dev.sh "$GPU" "$s" "$N" "$DIR/dev"
done
echo SEQ_DONE
