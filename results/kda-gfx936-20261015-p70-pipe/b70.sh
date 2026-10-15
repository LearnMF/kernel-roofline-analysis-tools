#!/bin/bash
# P-70 on bw46: build, bitwise gate with KDA_G2_PIPE=1 (+ several k), wall-time A/B of the forward
K=/public/home/tanbo/kra/examples/kda; T=/opt/kda_env/g2_p70; L=/public/home/tanbo/codex_validation/locks/bw46
cd /opt/kda_env && rm -rf g2_p70 && cp -a g2_r5 g2_p70 && cd g2_p70 && rm -rf build
python3 $K/p70_fwd_pipe.py . || exit 1
python3 build_g2.py > /tmp/b70.log 2>&1 && echo build_ok || { grep -m8 -E " error" /tmp/b70.log; exit 1; }
exec 9>$L/hcu_6.lock; flock -w 900 9 || exit 75
export KDA_G2=1 HIP_VISIBLE_DEVICES=6
r=$(KDA_G2_PIPE=1 CACHE=/tmp/gate_b70 bash $K/gate_all.sh $T "8192:12 8192:48" 2>&1)
echo "$r" | grep -E "PASS|FAIL" | tr '\n' ' '; echo
G=$K/bitwise_gate.py; export PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover
for kk in 1 8 64 127; do KDA_G2_PIPE=1 KDA_G2_PIPE_K=$kk python3 $G dump --tree $T --T 8192 --H 12 --out /tmp/gate_b70/k$kk.pt > /tmp/gate_b70/k$kk.log 2>&1 || tail -2 /tmp/gate_b70/k$kk.log
  echo "k=$kk: $(python3 $G cmp /tmp/gate_b70/base_8192_12.pt /tmp/gate_b70/k$kk.pt | tail -1)"; done
[[ "$r" == *"GATE_ALL PASS"* ]] || { echo "GATE FAILED -> no timing"; exit 1; }
cd /tmp
for TH in "16384 12" "65536 12"; do set -- $TH; python3 $K/pipe_time.py --tree $T --T $1 --H $2 --reps 20 2>&1 | grep "T="; done
for kk in 24 48 72; do echo "K=$kk: $(KDA_G2_PIPE_K=$kk python3 $K/pipe_time.py --tree $T --T 16384 --H 12 --reps 15 2>&1 | grep T=)"; done
echo P70_DONE
