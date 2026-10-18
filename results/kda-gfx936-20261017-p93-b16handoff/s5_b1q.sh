#!/bin/bash
# S5 fp64 oracle (frozen PR #556 reference_kda.py) for option B1' (P-93 --qk-only) vs production vs FLA, on bw46
K=/public/home/tanbo/kra/examples/kda; L=/public/home/tanbo/codex_validation/locks/bw46; G=${1:-3}
S5=/opt/kda_env/g2_r5/tests/s5/run_fp64_oracle.py
mkdir -p /tmp/s5_ref /tmp/s5_oracle && cp /public/home/tanbo/g2_opt/tests/interface/pr556/upstream/pr556/reference_kda.py.source /tmp/s5_ref/reference_kda.py
sha256sum /tmp/s5_ref/reference_kda.py | cut -c1-16
cd /tmp
PYTHONPATH=/opt/kda_env:/opt/kda_env/g2_r5:/opt/kda_env/takeover python3 $S5 --phase control 2>&1 | tail -1
CASES="8192:12 8192:48 16384:12 32768:4"
for c in $CASES; do IFS=: read -r T H <<<"$c"
  [ -f /tmp/s5_oracle/${T}_${H}_20260929.pt ] || \
    (PYTHONPATH=/opt/kda_env:/opt/kda_env/g2_r5:/opt/kda_env/takeover S5_THREADS=4 python3 $S5 --phase ref $T $H > /tmp/s5_ref_${T}_${H}.log 2>&1 &)
done
while pgrep -f "run_fp64_oracle.py --phase ref" > /dev/null; do sleep 20; done
for c in $CASES; do IFS=: read -r T H <<<"$c"; tail -1 /tmp/s5_ref_${T}_${H}.log 2>/dev/null; done
( exec 9>$L/hcu_$G.lock; flock -w 3600 9 || exit 75; export KDA_G2=1 HIP_VISIBLE_DEVICES=$G
  for c in $CASES; do IFS=: read -r T H <<<"$c"
    echo "== $T/$H production + FLA"
    PYTHONPATH=/opt/kda_env:/opt/kda_env/g2_r5:/opt/kda_env/takeover python3 $S5 --phase dut $T $H --arms g2,fla 2>/dev/null | grep -E "s5_dut|S5_FP64"
    echo "== $T/$H B1' (P-93 --qk-only)"
    PYTHONPATH=/opt/kda_env:/opt/kda_env/g2_p93q:/opt/kda_env/takeover python3 $S5 --phase dut $T $H --arms g2 2>/dev/null | grep -E "s5_dut|S5_FP64"
  done )
echo S5_B1Q_DONE
