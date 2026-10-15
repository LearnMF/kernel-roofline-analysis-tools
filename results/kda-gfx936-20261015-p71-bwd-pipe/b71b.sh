#!/bin/bash
K=/public/home/tanbo/kra/examples/kda; T=/opt/kda_env/g2_p71; L=/public/home/tanbo/codex_validation/locks/bw46
cd /opt/kda_env && rm -rf g2_p71 && cp -a g2_r5 g2_p71 && cd g2_p71 && rm -rf build
python3 $K/p71_rec_dav_pipe.py . || exit 1
python3 build_g2.py > /tmp/b71.log 2>&1 && echo build_ok || { grep -m8 -E " error" /tmp/b71.log; exit 1; }
exec 9>$L/hcu_6.lock; flock -w 900 9 || exit 75
export KDA_G2=1 HIP_VISIBLE_DEVICES=6 PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover
cd /tmp
python3 $K/p71_parts.py --tree $T 2>&1 | tail -16
G=$K/bitwise_gate.py
KDA_G2_BPIPE=1 python3 $G dump --tree $T --T 8192 --H 12 --out /tmp/gate_b71/cm2.pt > /tmp/gate_b71/cm2.log 2>&1 || tail -2 /tmp/gate_b71/cm2.log
echo "cm2 default m: $(python3 $G cmp /tmp/gate_b71/base_8192_12.pt /tmp/gate_b71/cm2.pt | tail -1)"
for mm in 96 128 160 192; do for pb in 16 32; do echo "16K M=$mm PUB=$pb: $(KDA_G2_BPIPE_PUB=$pb KDA_G2_BPIPE_M=$mm python3 $K/pipe_time.py --tree $T --T 16384 --H 12 --env KDA_G2_BPIPE --reps 12 2>&1 | grep T=)"; done; done
echo P71B_DONE
