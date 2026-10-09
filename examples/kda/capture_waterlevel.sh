#!/bin/bash
# P1 capture for the KDA operator: launcher interface trace + PMC (read/write presets).
# Run INSIDE the KDA container; analysis (`kra analyze`) can run anywhere afterwards.
#
#   KRA=/path/to/kernel-roofline-analysis-tools OUT=/tmp/kra_wl \
#   bash capture_waterlevel.sh "g2:8192:12 g2:8192:48"
set -euo pipefail
KRA=${KRA:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT=${OUT:-/tmp/kra_wl}
TREE=${TREE:-/opt/kda_env/g2_r5}
mkdir -p "$OUT"
for spec in ${1:-"g2:8192:12"}; do
  IFS=: read -r arm T H <<<"$spec"
  tag="${arm}_${T}_${H}"
  python3 "$KRA/examples/kda/kda_iter.py" --tree "$TREE" --arm "$arm" --T "$T" --H "$H" \
      --iters 2 --optrace "$OUT/$tag.optrace.json" > "$OUT/$tag.optrace.log" 2>&1
  (cd "$KRA" && python3 -m kra pmc --out "$OUT/$tag" --marker spin_kernel -- \
      python3 "$KRA/examples/kda/kda_iter.py" --tree "$TREE" --arm "$arm" --T "$T" --H "$H" --iters 1)
  echo "$tag done"
done
