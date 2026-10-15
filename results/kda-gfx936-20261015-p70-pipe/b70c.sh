#!/bin/bash
# P-70b on bw46: CU-masked overlap
K=/public/home/tanbo/kra/examples/kda; T=/opt/kda_env/g2_p70b; L=/public/home/tanbo/codex_validation/locks/bw46
cd /opt/kda_env && rm -rf g2_p70b && cp -a g2_r5 g2_p70b && cd g2_p70b && rm -rf build
python3 $K/p70_fwd_pipe.py . && python3 $K/p70b_cumask.py . || exit 1
python3 build_g2.py > /tmp/b70c.log 2>&1 && echo build_ok || { grep -m8 -E " error" /tmp/b70c.log; exit 1; }
exec 9>$L/hcu_6.lock; flock -w 900 9 || exit 75
export KDA_G2=1 HIP_VISIBLE_DEVICES=6 KDA_G2_PIPE_MASK=1
r=$(KDA_G2_PIPE=1 CACHE=/tmp/gate_b70c bash $K/gate_all.sh $T "8192:12" 2>&1)
echo "$r" | grep -E "PASS|FAIL" | tr '\n' ' '; echo
[[ "$r" == *"GATE_ALL PASS"* ]] || { echo "GATE FAILED -> no timing"; exit 1; }
export PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover; cd /tmp
for kk in 24 48 72 96; do echo "K=$kk: $(KDA_G2_PIPE_K=$kk python3 $K/pipe_time.py --tree $T --T 16384 --H 12 --reps 15 2>&1 | grep T=)"; done
for rc in 40 56; do echo "RCU=$rc K=48: $(KDA_G2_PIPE_RCU=$rc KDA_G2_PIPE_K=48 python3 $K/pipe_time.py --tree $T --T 16384 --H 12 --reps 15 2>&1 | grep T=)"; done
for kk in 96 192 288; do echo "64K K=$kk: $(KDA_G2_PIPE_K=$kk python3 $K/pipe_time.py --tree $T --T 65536 --H 12 --reps 10 2>&1 | grep T=)"; done
echo P70C_DONE
