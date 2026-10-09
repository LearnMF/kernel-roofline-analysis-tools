#!/bin/bash
# Bitwise gate of a candidate tree against the R5 production tree on the shape matrix,
# normal + keep-state paths.  Baseline dumps are cached in $CACHE.
#   bash gate_all.sh CAND_TREE ["8192:12 8192:48"]
set -uo pipefail
CAND=$1; SHAPES=${2:-"8192:12 8192:48"}
BASE=${BASE:-/opt/kda_env/g2_r5}; CACHE=${CACHE:-/tmp/gate}
KRA=${KRA:-$(cd "$(dirname "$0")/../.." && pwd)}; G=$KRA/examples/kda/bitwise_gate.py
mkdir -p "$CACHE"; fail=0
for s in $SHAPES; do
  T=${s%:*}; H=${s#*:}
  for ks in "" "--keep-state" "--gate-extreme"; do
    tag="${T}_${H}${ks:+_${ks#--}}"
    [ -f "$CACHE/base_$tag.pt" ] || PYTHONPATH=/opt/kda_env:$BASE:/opt/kda_env/takeover \
      python3 "$G" dump --tree "$BASE" --T "$T" --H "$H" --out "$CACHE/base_$tag.pt" $ks > /dev/null 2>&1
    PYTHONPATH=/opt/kda_env:$CAND:/opt/kda_env/takeover \
      python3 "$G" dump --tree "$CAND" --T "$T" --H "$H" --out "$CACHE/cand_$tag.pt" $ks > /dev/null 2>&1 \
      || { echo "$tag: candidate run FAILED"; fail=1; continue; }
    r=$(python3 "$G" cmp "$CACHE/base_$tag.pt" "$CACHE/cand_$tag.pt" | tail -1)
    echo "$tag: $r"; [[ "$r" == *PASS* ]] || { fail=1; python3 "$G" cmp "$CACHE/base_$tag.pt" "$CACHE/cand_$tag.pt"; }
  done
done
echo "GATE_ALL $([ $fail = 0 ] && echo PASS || echo FAIL)"; exit $fail
