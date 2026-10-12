#!/bin/bash
# B-class precision gate (R23b) for numerics-changing candidates (fusion / recompute / chunk size).
#   bash bclass_gate.sh <candidate tree> <out.jsonl> [gpu] [nseeds]
# Runs, on identical inputs, production (/opt/kda_env/g2_r5), the candidate and FLA (same BW) against
# the frozen FP64 oracle (reference_kda.py via tests/s5/run_fp64_oracle.py):
#   * nseeds paired seeds x {k3, extreme} gates at T=4096 H=12  (statistics)
#   * fixed long/wide cases: k3 8192x48, k3 16384x12, k3 32768x4, extreme 8192x12, extreme 4096x24
# Then bclass_report.py applies the four-rule gate's rules 1-3 (rule 4 = Megatron loss, separate).
CAND=$1; OUT=$2; GPU=${3:-1}; NS=${4:-6}
PROD=/opt/kda_env/g2_r5
export S5_REF=/opt/kda_env/reference_kda.py S5_TREE=/public/home/tanbo/g2_opt KDA_G2=1 HIP_VISIBLE_DEVICES=$GPU S5_THREADS=4
X=/public/home/tanbo/kra/examples/kda/s5_oracle_ext.py
C=/public/home/tanbo/s5work/oracle
cd /tmp; : > $OUT
cases=""
for s in $(seq 201 $((200 + NS))); do cases="$cases k3:4096:12:$s extreme:4096:12:$s"; done
cases="$cases k3:8192:48:20260929 k3:16384:12:7 k3:32768:4:11 extreme:8192:12:99 extreme:4096:24:5"
for spec in $cases; do
  IFS=: read -r gate T H seed <<< "$spec"
  S5_GATE=$gate S5_CACHE=$C python3 $X --phase ref $T $H $seed > /dev/null 2>&1   # no-op if cached
  for tree in $PROD $CAND; do
    arms=g2; [ "$tree" = "$PROD" ] && arms=g2,fla
    line=$(S5_GATE=$gate S5_CACHE=$C PYTHONPATH=/opt/kda_env:$tree:/opt/kda_env/takeover python3 $X --phase dut $T $H $seed --arms $arms 2>&1 | grep "\"tag\": \"s5_dut\"")
    echo "{\"tree\": \"$tree\", \"role\": \"$([ "$tree" = "$PROD" ] && echo prod || echo cand)\", \"gate\": \"$gate\", \"T\": $T, \"H\": $H, \"seed\": $seed, \"res\": ${line:-null}}" >> $OUT
  done
done
echo BCLASS_RUNS_DONE
