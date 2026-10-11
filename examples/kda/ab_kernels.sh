#!/bin/bash
# Per-kernel A/B: alternating hipprof traces (A,B,A,B,...) of the KDA fwd+bwd window,
# each with ITERS marker-delimited windows; prints per-kernel median device time of A and B
# (pooled over all windows of all rounds) and the ratio.  Device time is insensitive to
# host noise, which dominates end-to-end A/B at the 1% level.
#   bash ab_kernels.sh TREE_A TREE_B T H [ROUNDS=2] [ITERS=8] [OUT=/tmp/kra_abk]
#   KEEP_STATE=1: time the reentrant-checkpoint (Megatron) form instead of the plain fwd+bwd
#   JITTER=1: per-round placement jitter (R20) -- both sides sample the same placements
set -euo pipefail
A=$1; B=$2; T=$3; H=$4; ROUNDS=${5:-2}; ITERS=${6:-8}; OUT=${7:-/tmp/kra_abk}
KRA=${KRA:-$(cd "$(dirname "$0")/../.." && pwd)}
KS=${KEEP_STATE:+--keep-state}
PP_PRE=${PP_PRE:-/opt/kda_env}; PP_POST=${PP_POST:-/opt/kda_env/takeover}   # bw52: xplat/bwenv, stage_g2_bw7/takeover
mkdir -p "$OUT"; tag="${T}_${H}${KS:+_ks}"
trace() {  # tree label round
  KRA_JITTER_SEED=${JITTER:+$3} PYTHONPATH=$PP_PRE:$1:$PP_POST hipprof --hip-trace --output-type 0 \
    -o "$OUT/${2}_${tag}_r$3" python3 "$KRA/examples/kda/kda_iter.py" --tree "$1" --T "$T" --H "$H" \
    --iters "$ITERS" $KS > "$OUT/${2}_${tag}_r$3.log" 2>&1
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
# Kernels on ONE side only (a fusion removed / added a launch): their time belongs in the
# operator total -- the common-kernel sum alone would hide a removed kernel's saving.
oa = {k: st.median(v) for k, v in a.items() if k not in b}
ob = {k: st.median(v) for k, v in b.items() if k not in a}
for k, v in oa.items():
    print(f"{k:34s} {v:9.1f} {'—':>9s}   (A only)")
for k, v in ob.items():
    print(f"{k:34s} {'—':>9s} {v:9.1f}   (B only)")
print(f"{'TOTAL common kernels':34s} {ta:9.1f} {tb:9.1f} {ta/tb:7.4f}")
TA, TB = ta + sum(oa.values()), tb + sum(ob.values())
print(f"{'TOTAL (sum of medians)':34s} {TA:9.1f} {TB:9.1f} {TA/TB:7.4f}   (all kernels of each side)")
EOF
