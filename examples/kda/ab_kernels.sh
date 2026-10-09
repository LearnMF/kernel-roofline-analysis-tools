#!/bin/bash
# Per-kernel A/B: alternating hipprof traces (A,B,A,B,...) of the KDA fwd+bwd window,
# each with ITERS marker-delimited windows; prints per-kernel median device time of A and B
# (pooled over all windows of all rounds) and the ratio.  Device time is insensitive to
# host noise, which dominates end-to-end A/B at the 1% level.
#   bash ab_kernels.sh TREE_A TREE_B T H [ROUNDS=2] [ITERS=8] [OUT=/tmp/kra_abk]
set -euo pipefail
A=$1; B=$2; T=$3; H=$4; ROUNDS=${5:-2}; ITERS=${6:-8}; OUT=${7:-/tmp/kra_abk}
KRA=${KRA:-$(cd "$(dirname "$0")/../.." && pwd)}
mkdir -p "$OUT"; tag="${T}_${H}"
trace() {  # tree label round
  PYTHONPATH=/opt/kda_env:$1:/opt/kda_env/takeover hipprof --hip-trace --output-type 0 \
    -o "$OUT/${2}_${tag}_r$3" python3 "$KRA/examples/kda/kda_iter.py" --tree "$1" --T "$T" --H "$H" \
    --iters "$ITERS" > "$OUT/${2}_${tag}_r$3.log" 2>&1
}
for ((r = 0; r < ROUNDS; r++)); do
  if (( r % 2 == 0 )); then trace "$A" A $r; trace "$B" B $r; else trace "$B" B $r; trace "$A" A $r; fi
done
cd "$KRA" && python3 - "$OUT" "$tag" "$ROUNDS" <<'EOF'
import statistics as st, sys
from kra.timeline.hipprof_json import load
from kra.timeline.analyze import split_windows, short_name
out, tag, rounds = sys.argv[1], sys.argv[2], int(sys.argv[3])
def per_kernel(label):
    """kernel -> list (one per process/round) of per-process median times.  Buffer placement
    is fixed within a process and moves kernel time by several %, so the PROCESS is the
    independent sample, not the window."""
    d = {}
    for r in range(rounds):
        cur = {}
        for w in split_windows(load(f"{out}/{label}_{tag}_r{r}.json"), "spin_kernel", None):
            occ = {}
            for o in w:
                if o.kind != "kernel":
                    continue
                n = short_name(o.name); occ[n] = occ.get(n, 0) + 1
                cur.setdefault(f"{n}#{occ[n]}", []).append(o.dur_ns / 1e3)
        for k, v in cur.items():
            d.setdefault(k, []).append(st.median(v))
    return d
a, b = per_kernel("A"), per_kernel("B")
print(f"{'kernel':34s} {'A us':>9s} {'B us':>9s} {'A/B':>7s}  {'A spread':>8s} {'B spread':>8s}  procs  (median of per-process medians; spread = (max-min)/median)")
ta = tb = 0.0
for k in a:
    if k not in b:
        continue
    ma, mb = st.median(a[k]), st.median(b[k]); ta += ma; tb += mb
    sa, sb = (max(a[k]) - min(a[k])) / ma, (max(b[k]) - min(b[k])) / mb
    print(f"{k:34s} {ma:9.1f} {mb:9.1f} {ma/mb:7.4f}  {sa:8.2%} {sb:8.2%}  {len(a[k])}/{len(b[k])}")
print(f"{'TOTAL (sum of medians)':34s} {ta:9.1f} {tb:9.1f} {ta/tb:7.4f}")
EOF
