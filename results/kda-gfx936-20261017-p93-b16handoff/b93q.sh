#!/bin/bash
# P-93 (option B1, not bitwise) on bw46: bf16 s4b->tail handoff -- accuracy vs the fp32 reference and per-kernel A/B
K=/public/home/tanbo/kra/examples/kda; X=/public/home/tanbo/xplat; L=/public/home/tanbo/codex_validation/locks/bw46; G=${1:-4}
cd /opt/kda_env && rm -rf g2_p93q && cp -a g2_r5 g2_p93q && cd g2_p93q && rm -rf build
python3 $K/p93_b16_handoff.py . --qk-only || exit 1
python3 build_g2.py > /tmp/b93q.log 2>&1 && echo build_ok || { grep -m12 -B2 -A3 error /tmp/b93.log | head -40; exit 1; }
( exec 9>$L/hcu_$G.lock; flock -w 3600 9 || exit 75
  export KDA_G2=1 HIP_VISIBLE_DEVICES=$G; cd /tmp
  for TH in "4096 12" "8192 12" "4096 48"; do set -- $TH
    for tr in g2_p93q; do
      echo "== acc $tr T=$1 H=$2"
      PYTHONPATH=/opt/kda_env:/opt/kda_env/$tr:/opt/kda_env/takeover timeout 1800 python3 /opt/kda_env/$tr/tests/g2/fp32_ref_check.py $1 $2 2>/tmp/b93q_acc.err | grep -E "s2fp32|S2_FP32" || tail -3 /tmp/b93_acc.err
    done
  done
  export KDA_SKIP_FLA_PRUNE=1 JITTER=1
  for TH in "16384 12" "16384 48"; do set -- $TH
    bash $K/ab_kernels.sh /opt/kda_env/g2_r5 /opt/kda_env/g2_p93q $1 $2 5 6 /tmp/abk93q_$G > $X/b46_p93q_$1_$2.txt 2>&1
    echo "== perf $1/$2 (A = production fp32 handoff, B = bf16 handoff)"; grep -E "s4b|tail|TOTAL common" $X/b46_p93q_$1_$2.txt; done )
echo P93Q_DONE
