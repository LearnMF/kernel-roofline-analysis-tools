#!/bin/bash
K=/public/home/tanbo/kra/examples/kda; T=/opt/kda_env/g2_p71; L=/public/home/tanbo/codex_validation/locks/bw46
exec 9>$L/hcu_6.lock; flock -w 900 9 || exit 75
export KDA_G2=1 HIP_VISIBLE_DEVICES=6 PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover KDA_G2_BPIPE=1 KDA_G2_BPIPE_M=${1:-192}
D=/tmp/p71tr; rm -rf $D; mkdir -p $D; cd $D
hipprof --hip-trace --output-type 0 -o $D/x python3 $K/kda_iter.py --tree $T --T 16384 --H 12 --iters 2 --warmup 1 > log.txt 2>&1
python3 - <<'PY'
import json
d=json.load(open("/tmp/p71tr/x.json"))
ev=d["traceEvents"] if isinstance(d,dict) else d
ks=[e for e in ev if e.get("ph")=="X" and e.get("pid")==3]
ks.sort(key=lambda e:e["ts"])
wu=[k for k,e in enumerate(ks) if "g2_wu_kernel" in e["name"]]
a=wu[-1]
t0=ks[a]["ts"]
for e in ks[a:a+12]:
    print(round(e["ts"]-t0,1), round(e["dur"],1), e.get("tid"), e["name"][:70])
PY
