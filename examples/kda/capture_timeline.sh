#!/bin/bash
# L1 capture for the KDA operator: unprofiled reference timing + hipprof trace + analysis.
# Run INSIDE the KDA container (needs torch, hip_kda, hipprof).
#
#   KRA=/path/to/kernel-roofline-analysis-tools MACHINE=machines/gfx936-bw1000.json \
#   OUT=/tmp/kra_tl bash capture_timeline.sh "g2:8192:12 g2:8192:48 fla:8192:12"
#
# Env expected by the KDA bench (example for the tanbo_mega_k3 container):
#   KDA_G2=1 PYTHONPATH=/opt/kda_env:/opt/kda_env/g2_r5:/opt/kda_env/takeover
set -euo pipefail
KRA=${KRA:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT=${OUT:-/tmp/kra_tl}
TREE=${TREE:-/opt/kda_env/g2_r5}
MACHINE=${MACHINE:-}
ITERS_REF=${ITERS_REF:-20}
ITERS_TRACE=${ITERS_TRACE:-5}
mkdir -p "$OUT"
for spec in ${1:-"g2:8192:12"}; do
  IFS=: read -r arm T H <<<"$spec"
  tag="${arm}_${T}_${H}"
  ref=$(python3 "$KRA/examples/kda/kda_iter.py" --tree "$TREE" --arm "$arm" --T "$T" --H "$H" \
          --iters "$ITERS_REF" --time 2>/dev/null | sed -n 's/^KDA_ITER_MS median=\([0-9.]*\).*/\1/p')
  rm -f "$OUT/$tag".{db,json,hipkernel.csv,hiptrace.csv}
  hipprof --hip-trace --output-type 0 -o "$OUT/$tag" \
    python3 "$KRA/examples/kda/kda_iter.py" --tree "$TREE" --arm "$arm" --T "$T" --H "$H" \
      --iters "$ITERS_TRACE" > "$OUT/$tag.hipprof.log" 2>&1
  (cd "$KRA" && python3 -m kra timeline "$OUT/$tag.json" --marker spin_kernel \
      ${MACHINE:+--machine "$MACHINE"} ${ref:+--reference-ms "$ref"} \
      --title "KDA $arm T=$T H=$H fwd+bwd (L1)")
  echo "$tag reference_ms=$ref"
done
