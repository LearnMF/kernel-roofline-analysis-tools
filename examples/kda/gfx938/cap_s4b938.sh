#!/bin/bash
# SQTT of p3_wy_s4b_kernel on gfx938 (nmz5): cap_s4b938.sh T H TAG
T=${1:-2048}; H=${2:-16}; TAG=${3:-s4b}
source /opt/dtk/env.sh >/dev/null 2>&1
export HIP_VISIBLE_DEVICES=6 PYTHONPATH=/public/home/tanbo/ft938_inst
D=/public/home/tanbo/sqtt938/${TAG}_${T}_${H}; rm -rf $D; mkdir -p $D; cd /tmp
hipprof --sqtt --sqtt-type 1 --kernel-name p3_wy_s4b_kernel --output-type 0 -o $D/x \
   python3 /public/home/tanbo/mk3/ft_card_run.py --T $T --H $H --iters 1 --warmup 1 > $D/log.txt 2>&1
echo rc=$?; ls -la $D | head -20; tail -5 $D/log.txt
