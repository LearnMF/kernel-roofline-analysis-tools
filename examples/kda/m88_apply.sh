#!/bin/bash
# apply the P-88 merge (cwd = the dir holding g2_wy.cuh): s4a / s4b on the head-fastest grid by default
set -e
python3 /public/home/tanbo/kra/examples/kda/p88_s4_hf.py .
sed -i 's/g2::g2_env_long("KDA_G2_S4_HF", 0) ? dim3(H, NT) : dim3(NT, H);   \/\/ kra P-88/g2::g2_env_long("KDA_G2_S4_HF", 1) ? dim3(H, NT) : dim3(NT, H);   \/\/ kra P-88 (default on)/g' g2_wy.cuh
test "$(grep -c 'KDA_G2_S4_HF", 1' g2_wy.cuh)" = 2 && echo "P-88 default on"
