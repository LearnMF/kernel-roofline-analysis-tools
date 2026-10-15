#!/bin/bash
# P-71 on bw46: recompute || dav part 1 (consumer-side pipeline)
K=/public/home/tanbo/kra/examples/kda; T=/opt/kda_env/g2_p71; L=/public/home/tanbo/codex_validation/locks/bw46
cd /opt/kda_env && rm -rf g2_p71 && cp -a g2_r5 g2_p71 && cd g2_p71 && rm -rf build
python3 $K/p71_rec_dav_pipe.py . || exit 1
python3 build_g2.py > /tmp/b71.log 2>&1 && echo build_ok || { grep -m8 -E " error" /tmp/b71.log; exit 1; }
exec 9>$L/hcu_6.lock; flock -w 900 9 || exit 75
export KDA_G2=1 HIP_VISIBLE_DEVICES=6
r=$(KDA_G2_BPIPE=1 CACHE=/tmp/gate_b71 bash $K/gate_all.sh $T "8192:12 8192:48" 2>&1)
echo "$r" | grep -E "PASS|FAIL" | tr '\n' ' '; echo
G=$K/bitwise_gate.py; export PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover
for mm in 0 64 128; do KDA_G2_BPIPE=1 KDA_G2_BPIPE_M=$mm python3 $G dump --tree $T --T 8192 --H 12 --out /tmp/gate_b71/m$mm.pt > /tmp/gate_b71/m$mm.log 2>&1 || tail -2 /tmp/gate_b71/m$mm.log
  echo "m=$mm: $(python3 $G cmp /tmp/gate_b71/base_8192_12.pt /tmp/gate_b71/m$mm.pt | tail -1)"; done
[[ "$r" == *"GATE_ALL PASS"* ]] || { echo "GATE FAILED -> no timing"; exit 1; }
cd /tmp
for mm in 128 192 230 256; do echo "16K M=$mm: $(KDA_G2_BPIPE_M=$mm python3 $K/pipe_time.py --tree $T --T 16384 --H 12 --env KDA_G2_BPIPE --reps 15 2>&1 | grep T=)"; done
for mm in 768 920; do echo "64K M=$mm: $(KDA_G2_BPIPE_M=$mm python3 $K/pipe_time.py --tree $T --T 65536 --H 12 --env KDA_G2_BPIPE --reps 8 2>&1 | grep T=)"; done
echo P71_DONE
