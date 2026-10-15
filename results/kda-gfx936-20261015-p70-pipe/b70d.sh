#!/bin/bash
# P-70b on bw46: CU-masked overlap
K=/public/home/tanbo/kra/examples/kda; T=/opt/kda_env/g2_p70c; L=/public/home/tanbo/codex_validation/locks/bw46
cd /opt/kda_env && rm -rf g2_p70c && cp -a g2_r5 g2_p70c && cd g2_p70c && rm -rf build
python3 $K/p70_fwd_pipe.py . && python3 $K/p70b_cumask.py . && python3 $K/p70c_tailcm.py . || exit 1
python3 build_g2.py > /tmp/b70d.log 2>&1 && echo build_ok || { grep -m8 -E " error" /tmp/b70d.log; exit 1; }
exec 9>$L/hcu_6.lock; flock -w 900 9 || exit 75
export KDA_G2=1 HIP_VISIBLE_DEVICES=6 KDA_G2_PIPE_MASK=1
r=$(KDA_G2_PIPE=1 CACHE=/tmp/gate_b70d bash $K/gate_all.sh $T "8192:12" 2>&1)
echo "$r" | grep -E "PASS|FAIL" | tr '\n' ' '; echo
[[ "$r" == *"GATE_ALL PASS"* ]] || { echo "GATE FAILED -> no timing"; exit 1; }
export PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover; cd /tmp
for kk in 16 32 48 64; do echo "K=$kk: $(KDA_G2_PIPE_K=$kk python3 $K/pipe_time.py --tree $T --T 16384 --H 12 --reps 15 2>&1 | grep T=)"; done
for rc in 40 56; do echo "RCU=$rc K=48: $(KDA_G2_PIPE_RCU=$rc KDA_G2_PIPE_K=48 python3 $K/pipe_time.py --tree $T --T 16384 --H 12 --reps 15 2>&1 | grep T=)"; done
for kk in 128 192 256; do echo "64K K=$kk: $(KDA_G2_PIPE_K=$kk python3 $K/pipe_time.py --tree $T --T 65536 --H 12 --reps 10 2>&1 | grep T=)"; done
echo P70D_DONE
