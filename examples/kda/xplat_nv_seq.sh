#!/bin/bash
# Clean A800 re-run inside the NV container: all shapes SEQUENTIAL on one GPU, N processes each.
#   bash xplat_nv_seq.sh GPU OUT [N=3]
GPU=$1; OUT=$2; N=${3:-3}
for s in 8192:12 8192:24 8192:48 8192:96 16384:12 32768:12; do
  bash /xplat/xplat_nv.sh "$GPU" "$s" "$N" "$OUT"
done
echo SEQ_DONE
