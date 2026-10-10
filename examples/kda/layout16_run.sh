#!/bin/bash
# Build layout16_bench from the production sources (base) and from the paired-load prototype
# (proto), then run base/proto alternately in separate processes (placement noise is per
# process) and print per-kernel medians.   bash layout16_run.sh TREE [ROUNDS=6]
set -euo pipefail
TREE=$1; ROUNDS=${2:-6}; KRA=${KRA:-$(cd "$(dirname "$0")/../.." && pwd)}
W=/tmp/kra_l16; rm -rf $W; mkdir -p $W/base $W/proto
cp -r "$TREE/csrc/g2/." $W/base/; cp -r "$TREE/csrc/g2/." $W/proto/
python3 "$KRA/examples/kda/layout16_proto.py" $W/proto
for v in base proto; do
  cp "$KRA/examples/kda/layout16_bench.hip" $W/$v/
  (cd $W/$v && /opt/dtk/bin/hipcc -O3 -std=c++17 --offload-arch=gfx936 -I. -DHIP_ENABLE_WARP_SYNC_BUILTINS=1 \
     layout16_bench.hip -o $W/bench_$v 2>&1 | grep -E " error" || true)
  /opt/dtk/bin/hipcc -O3 -std=c++17 --offload-arch=gfx936 -I$W/$v -DHIP_ENABLE_WARP_SYNC_BUILTINS=1 \
     --cuda-device-only -S $W/$v/layout16_bench.hip -o $W/$v.s 2>/dev/null || true
done
for v in base proto; do
  for k in _ZN2g220g2_fwd_h_v8o4_kernelILb0EEEvNS_10FwdOParamsE g2_dhu_v8b_kernel _ZN2g218g2_dhu_v8b4_kernelENS_10BwdNParamsE; do
    n2=$(awk -v s="$k:" '$1==s{f=1} f{print} f&&/^.Lfunc_end/{exit}' $W/$v.s | grep -c "global_load_dwordx2" || true)
    n4=$(awk -v s="$k:" '$1==s{f=1} f{print} f&&/^.Lfunc_end/{exit}' $W/$v.s | grep -c "global_load_dwordx4" || true)
    vg=$(awk -v s=".amdhsa_kernel $k" 'index($0,s){f=1} f&&/next_free_vgpr/{print $2; exit}' $W/$v.s)
    echo "ISA $v $k x2=$n2 x4=$n4 vgpr=$vg"
  done
done
for ((r = 0; r < ROUNDS; r++)); do
  for s in "8192 12" "8192 48"; do
    if (( r % 2 == 0 )); then o="base proto"; else o="proto base"; fi
    for v in $o; do $W/bench_$v $s 20 | sed "s/^/$v /"; done
  done
done > $W/runs.txt
python3 - $W/runs.txt <<'EOF'
import json, statistics as st, sys, collections
d = collections.defaultdict(list)
for ln in open(sys.argv[1]):
    v, js = ln.split(" ", 1)
    r = json.loads(js); d[(r["kernel"], r["T"], r["H"], v)].append(r["median_us"])
for (k, T, H, v) in sorted({(a, b, c, "base") for a, b, c, _ in d}):
    b, p = d[(k, T, H, "base")], d[(k, T, H, "proto")]
    mb, mp = st.median(b), st.median(p)
    print(f"{k:10s} {T}/{H:<3d} base {mb:8.1f} us  proto {mp:8.1f} us  base/proto {mb/mp:.4f}  "
          f"spread base {(max(b)-min(b))/mb:.2%} proto {(max(p)-min(p))/mp:.2%}  n={len(b)}")
EOF
