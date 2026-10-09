#!/bin/bash
# L3 SQTT capture of the production G2 kernels from a given source tree (run on the HOST
# of the node; uses the xprof container, which cannot see /public, so sources are copied in).
#
#   bash sqtt_capture.sh TREE TAG T H "kernel1,kernel2" [GPU=0]
# Produces /public/home/tanbo/kra_sqtt/TAG/<...>.perf (one perf per captured dispatch set).
# The bench (examples/kda/bn_sqtt.hip) launches every production kernel per repetition;
# rep 1 is warm-up, the dispatch filter selects rep 2.
set -euo pipefail
TREE=$1; TAG=$2; T=$3; H=$4; KERNELS=$5; GPU=${6:-0}
C=${XPROF_CONTAINER:-glm_bw7_xprof2}
KRA=${KRA:-/public/home/tanbo/kra}
OUT=/public/home/tanbo/kra_sqtt/$TAG; mkdir -p "$OUT"
W=/tmp/kra_sqtt_$TAG
docker exec "$C" rm -rf "$W"; docker exec "$C" mkdir -p "$W"
docker cp "$TREE/csrc/g2" "$C:$W/g2"
docker cp "$KRA/examples/kda/bn_sqtt.hip" "$C:$W/g2/bn_sqtt.hip"
docker exec "$C" bash -lc "cd $W/g2 && /opt/dtk/bin/hipcc -O3 -std=c++17 --offload-arch=gfx936 -I. \
  -DHIP_ENABLE_WARP_SYNC_BUILTINS=1 bn_sqtt.hip -o $W/bn_sqtt 2>&1 | grep -E ' error' || true"
# dispatch numbering: list all dispatches of a plain 2-rep run (no profiling sections)
docker exec -e HIP_VISIBLE_DEVICES=$GPU "$C" bash -lc "export PATH=/opt/rocm-6.3.3/bin:\$PATH; cd $W && \
  xprof --kernels ${KERNELS} --output-dir $W/list ./bn_sqtt $T $H 2 > $W/list.log 2>&1 || true"
RANGE=$(docker exec "$C" bash -lc "grep -o 'dispatch kernel \[[0-9]*\]' $W/list.log | grep -o '[0-9]*' | sort -n" \
  | python3 -c "import sys; d=[int(x) for x in sys.stdin.read().split()]; n=len(d)//2; print(f'{d[n]}:{d[-1]+1}')")
echo "dispatch range (rep 2): $RANGE"
docker exec -e HIP_VISIBLE_DEVICES=$GPU "$C" bash -lc "export PATH=/opt/rocm-6.3.3/bin:\$PATH; cd $W && \
  xprof --enable-sqtt --kernels ${KERNELS} --dispatches $RANGE --output-dir $W/sqtt ./bn_sqtt $T $H 2 > $W/sqtt.log 2>&1; \
  ls -la $W/sqtt"
docker cp "$C:$W/sqtt/." "$OUT/"
docker cp "$C:$W/sqtt.log" "$OUT/sqtt.log"
ls -la "$OUT"
