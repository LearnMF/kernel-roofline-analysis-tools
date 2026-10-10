#!/bin/bash
# Bitwise gate of a candidate tree against the R5 production tree on the shape matrix,
# normal + keep-state paths.  Baseline dumps are cached in $CACHE.
#   bash gate_all.sh CAND_TREE ["8192:12 8192:48"]
set -uo pipefail
CAND=$1; SHAPES=${2:-"8192:12 8192:48"}
BASE=${BASE:-/opt/kda_env/g2_r5}; CACHE=${CACHE:-/tmp/gate}
# PYTHONPATH around the tree (bw7 container: /opt/kda_env; bw52: PP_PRE=xplat/bwenv, PP_POST=stage_g2_bw7/takeover)
PP_PRE=${PP_PRE:-/opt/kda_env}; PP_POST=${PP_POST:-/opt/kda_env/takeover}
KRA=${KRA:-$(cd "$(dirname "$0")/../.." && pwd)}; G=$KRA/examples/kda/bitwise_gate.py
mkdir -p "$CACHE"; fail=0
for s in $SHAPES; do
  T=${s%:*}; H=${s#*:}
  for ks in "" "--keep-state" "--gate-extreme"; do
    tag="${T}_${H}${ks:+_${ks#--}}"
    [ -f "$CACHE/base_$tag.pt" ] || PYTHONPATH=$PP_PRE:$BASE:$PP_POST \
      python3 "$G" dump --tree "$BASE" --T "$T" --H "$H" --out "$CACHE/base_$tag.pt" $ks > /dev/null 2>&1
    PYTHONPATH=$PP_PRE:$CAND:$PP_POST \
      python3 "$G" dump --tree "$CAND" --T "$T" --H "$H" --out "$CACHE/cand_$tag.pt" $ks > /dev/null 2>&1 \
      || { echo "$tag: candidate run FAILED"; fail=1; continue; }
    r=$(python3 "$G" cmp "$CACHE/base_$tag.pt" "$CACHE/cand_$tag.pt" | tail -1)
    echo "$tag: $r"; [[ "$r" == *PASS* ]] || { fail=1; python3 "$G" cmp "$CACHE/base_$tag.pt" "$CACHE/cand_$tag.pt"; }
  done
done
# varlen: packed sequences with lengths that are not multiples of 64 (tail chunks, a short
# sequence, a 1-token sequence) -- exercises the VL template instantiations.
VARLEN=${VARLEN:-"8192:12:0,1000,1063,1064,5000,8192 8192:48:0,63,4097,8192"}
for s in $VARLEN; do
  IFS=: read -r T H CUS <<<"$s"
  for ks in "" "--keep-state"; do
    tag="vl_${T}_${H}_$(echo "$CUS" | md5sum | cut -c1-6)${ks:+_${ks#--}}"
    [ -f "$CACHE/base_$tag.pt" ] || PYTHONPATH=$PP_PRE:$BASE:$PP_POST \
      python3 "$G" dump --tree "$BASE" --T "$T" --H "$H" --cu "$CUS" --out "$CACHE/base_$tag.pt" $ks > "$CACHE/base_$tag.log" 2>&1 \
      || { echo "$tag: BASE run FAILED (see $CACHE/base_$tag.log)"; fail=1; continue; }
    PYTHONPATH=$PP_PRE:$CAND:$PP_POST \
      python3 "$G" dump --tree "$CAND" --T "$T" --H "$H" --cu "$CUS" --out "$CACHE/cand_$tag.pt" $ks > /dev/null 2>&1 \
      || { echo "$tag: candidate run FAILED"; fail=1; continue; }
    r=$(python3 "$G" cmp "$CACHE/base_$tag.pt" "$CACHE/cand_$tag.pt" | tail -1)
    echo "$tag ($CUS): $r"; [[ "$r" == *PASS* ]] || fail=1
  done
done
echo "GATE_ALL $([ $fail = 0 ] && echo PASS || echo FAIL)"; exit $fail
