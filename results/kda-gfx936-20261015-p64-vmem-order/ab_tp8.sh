#!/bin/bash
# TP8-only A/B (16K/12, 64K/12) of candidate trees vs g2_r5 on bw7 HCU 4/6/7 (HCU5 shared with another user)
X=/public/home/tanbo/xplat
run() { bash $X/abk7.sh /opt/kda_env/g2_r5 /opt/kda_env/$1 $2 12 $3 $X/b7_$4_$2_12.txt; }
( run g2_p64B 16384 4 p64B; run g2_p63c_b7 16384 4 p63c ) &
( run g2_p64B 65536 6 p64B ) &
( run g2_p63c_b7 65536 7 p63c ) &
wait; echo AB_TP8_DONE
