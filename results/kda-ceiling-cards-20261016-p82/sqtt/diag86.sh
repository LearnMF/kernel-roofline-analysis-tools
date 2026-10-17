#!/bin/bash
# SQTT round 2 on production (P-83/P-84b/P-85): s4a, s4b (TP8 + TP2), prep_a (TP8), wu (TP8)
X=/public/home/tanbo/xplat; T=/opt/kda_env/g2_r5; G=${1:-1}
bash $X/cap_k3.sh p3_wy_s4a_kernel r2_s4a_16k12 $G $T 16384 12 > /dev/null 2>&1; echo s4a_16k12
bash $X/cap_k3.sh p3_wy_s4a_kernel r2_s4a_8k48 $G $T 8192 48 > /dev/null 2>&1; echo s4a_8k48
bash $X/cap_k3.sh p3_wy_s4b_kernel r2_s4b_16k12 $G $T 16384 12 > /dev/null 2>&1; echo s4b_16k12
bash $X/cap_k3.sh p3_wy_s4b_kernel r2_s4b_8k48 $G $T 8192 48 > /dev/null 2>&1; echo s4b_8k48
bash $X/cap_k3.sh g2_fwd_prep_a_kernel r2_prepa_16k12 $G $T 16384 12 > /dev/null 2>&1; echo prepa_16k12
bash $X/cap_k3.sh g2_wu_kernel r2_wu_16k12 $G $T 16384 12 > /dev/null 2>&1; echo wu_16k12
echo DIAG86_DONE
