#!/bin/bash
# Interleaved A/B timing of the KDA fwd+bwd between two hip_kda trees (unprofiled).
# Each round runs A then B (order alternates every round), each run = median of ITERS
# iterations (kda_iter.py --time).  Prints per-round medians and the B/A ratio.
#
#   bash ab_time.sh TREE_A TREE_B T H [ROUNDS=6] [ITERS=20]
set -euo pipefail
A=$1; B=$2; T=$3; H=$4; ROUNDS=${5:-6}; ITERS=${6:-20}
KRA=${KRA:-$(cd "$(dirname "$0")/../.." && pwd)}
run() {  # tree -> median ms
  PYTHONPATH=/opt/kda_env:$1:/opt/kda_env/takeover python3 "$KRA/examples/kda/kda_iter.py" \
    --tree "$1" --T "$T" --H "$H" --iters "$ITERS" --time 2>/dev/null \
    | sed -n 's/^KDA_ITER_MS median=\([0-9.]*\).*/\1/p'
}
ra=(); rb=()
for ((i = 0; i < ROUNDS; i++)); do
  if (( i % 2 == 0 )); then a=$(run "$A"); b=$(run "$B"); else b=$(run "$B"); a=$(run "$A"); fi
  ra+=("$a"); rb+=("$b"); echo "round $i: A=$a B=$b"
done
python3 - "${ra[*]}" "${rb[*]}" <<'EOF'
import statistics as s, sys
a = [float(x) for x in sys.argv[1].split()]; b = [float(x) for x in sys.argv[2].split()]
ma, mb = s.median(a), s.median(b)
print(f"AB_RESULT A_median_ms={ma:.4f} B_median_ms={mb:.4f} speedup_A_over_B={ma/mb:.4f} "
      f"A_spread={max(a)-min(a):.4f} B_spread={max(b)-min(b):.4f}")
EOF
