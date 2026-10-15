#!/bin/bash
# P-64 variants on bw7: build A/B/C from g2_r5 (P-62 production), quick bitwise gate (8192:12 dense, normal/keep/extreme),
# SQTT critical path of the forward step (dispatch 2)
K=/public/home/tanbo/kra/examples/kda
for v in A B C; do
  T=/opt/kda_env/g2_p64$v
  ( cd /opt/kda_env && rm -rf g2_p64$v && cp -a g2_r5 g2_p64$v && cd g2_p64$v && rm -rf build && python3 $K/p64_vmem_order.py . $v > /dev/null && python3 build_g2.py > /tmp/p64$v.log 2>&1 && echo "$v build_ok" ) || { echo "$v build FAILED"; grep -m3 error /tmp/p64$v.log; continue; }
done
L=/public/home/tanbo/codex_validation/locks/bw7
for v in A B C; do
  T=/opt/kda_env/g2_p64$v
  r=$( exec 9>$L/hcu_4.lock; flock -w 900 9 || exit 75; KDA_G2=1 HIP_VISIBLE_DEVICES=4 CACHE=/tmp/gate_p64 VARLEN=" " bash $K/gate_all.sh $T "8192:12" 2>&1 )
  echo "$v: $(echo "$r" | grep -E "PASS|FAIL" | tr '\n' ' ')"
  [[ "$r" == *"GATE_ALL PASS"* ]] || continue
  bash /public/home/tanbo/xplat/cap_b7.sh $T p64$v > /dev/null 2>&1
  echo "$v fwd: $(grep -m1 'critical path per step' /public/home/tanbo/xplat/cp/cp_p64${v}_d2.txt)"
  grep -m1 "stall:vmem-load " /public/home/tanbo/xplat/cp/cp_p64${v}_d2.txt
done
echo P64_DONE
