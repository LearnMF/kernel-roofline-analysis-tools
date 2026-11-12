#!/bin/bash
# lb_pmc.sh VAR GPU SHAPE TAG [ENV=..]... : hipprof PMC (LDS counters) on kbench --iters 2; prints the last g2_dav / g2_wu rows
source /opt/dtk/env.sh >/dev/null 2>&1; cd /tmp; V=$1; G=$2; SH=$3; TAG=$4; shift 4
O=/tmp/lbp_${V}_$TAG; rm -rf $O; mkdir -p $O
echo "pmc: ${LB_PMC:-SQ_LDS_BANK_CONFLICT SQ_LDS_IDX_ACTIVE SQ_INSTS_LDS SQ_WAIT_INST_LDS}" > $O/in.txt
env "$@" FTRAIN_KDA_DEV=1 HIPPROF_INPUT_FILE=$O/in.txt HIP_VISIBLE_DEVICES=$G PYTHONPATH=/public/home/tanbo/b938/$V \
  hipprof --pmc --pmc-type 3 -o $O/ python3 /public/home/tanbo/mk3/kbench.py --shapes $SH --iters 2 > $O/log 2>&1
python3 - $O "$TAG" <<'PY'
import csv, glob, sys, json
rows = {}
for fn in glob.glob(sys.argv[1] + "/*.csv"):
    for r in csv.DictReader(open(fn)):
        for k in ("g2_dav", "g2_wu"):
            if k in r["KernelName"]: rows[k] = r
for k, r in sorted(rows.items()):
    out = {"tag": sys.argv[2], "k": k, "us": round((float(r["EndNs"]) - float(r["BeginNs"])) / 1e3, 1)}
    for c in r:
        if c.startswith("SQ_") or c.startswith("TCC") or c.startswith("TA_"): out[c] = float(r[c] or 0)
    print(json.dumps(out))
PY
