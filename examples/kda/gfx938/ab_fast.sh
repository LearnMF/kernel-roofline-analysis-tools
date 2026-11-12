#!/bin/bash
# ab_fast.sh KEY SHAPES GPU "name1=pythonpath1[:ENV=..]" "name2=..."  -> two interleaved rounds, one process per build per round
source /opt/dtk/env.sh >/dev/null 2>&1; cd /tmp; KEY=$1; SH=$2; G=$3; shift 3
M=/public/home/tanbo/mk3; OUT=/tmp/abf_$$.jsonl; rm -f $OUT
for round in 1 2; do for spec in "$@"; do
  name=${spec%%=*}; rest=${spec#*=}; pp=${rest%%:*}; envs=""; [ "$rest" != "$pp" ] && envs=${rest#*:}
  env FTRAIN_KDA_DEV=1 $envs HIP_VISIBLE_DEVICES=$G PYTHONPATH=$pp python3 $M/kbench.py --key $KEY --shapes $SH --iters 20 --tag $name 2>/dev/null | grep '^{' >> $OUT
done; done
python3 - $OUT <<'PY'
import json, sys, collections, statistics as st
rows = [json.loads(l) for l in open(sys.argv[1])]
by = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows: by[(r["T"], r["H"])][r["tag"]].append(r)
for s, d in sorted(by.items()):
    names = list(d); b = d[names[0]]
    kb = min(r["k_med"] for r in b); tb = min(r["tot_avg"] for r in b)
    line = f"{s}: {names[0]} k {kb:.1f} us tot {tb/1000:.3f} ms"
    for n in names[1:]:
        x = d[n]; line += f" | {n}: k {kb/min(r['k_med'] for r in x):.3f}x tot {tb/min(r['tot_avg'] for r in x):.3f}x (rounds k " + "/".join(f"{r['k_med']:.1f}" for r in x) + ")"
    print(line)
PY
