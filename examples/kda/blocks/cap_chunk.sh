#!/bin/bash
# SQTT of the chunk-parallel production kernels at 16K/12 -> one CTA's dynamic traces each
export KDA_G2=1 HIP_VISIBLE_DEVICES=4
PP=/public/home/tanbo/xplat/bwenv:/public/home/tanbo/g2_opt:/public/home/tanbo/codex_validation/stage_g2_bw7/takeover
OUT=/public/home/tanbo/xplat/blocks/cta_traces; mkdir -p $OUT
for spec in fwd_prep_a:4 fwd_prep_b:4 l2n_apply:4 gate_kernel:4 g2_wu_kernel:4 g2_dav_kernel:4 wy_s4a:4 wy_s4b:4 bwd_tail_kernel:4; do
  k=${spec%:*}; W=${spec#*:}; D=/tmp/sqc_$k; rm -rf $D; mkdir -p $D
  cd /tmp && PYTHONPATH=$PP hipprof --sqtt --sqtt-type 1 --kernel-name $k --output-type 0 -o $D/x \
     python3 /public/home/tanbo/kra/examples/kda/kda_iter.py --tree /public/home/tanbo/g2_opt --T 16384 --H 12 --iters 1 --warmup 1 > $D/log.txt 2>&1
  disp=$(ls $D | grep -oE "thread_trace_[0-9]+" | sort -u | tail -1 | grep -oE "[0-9]+$")
  python3 /public/home/tanbo/xplat/blocks/pk/cta_trace.py $D ${disp:-0} $W $OUT/$k.json 2>&1 | tail -1
done
