#!/bin/bash
# C-3 on bw46: CTA-count effect on the forward recurrence (v8o4w NVH=2: 48 CTAs vs NVH=1: 96 CTAs at H=12)
K=/public/home/tanbo/kra/examples/kda; T=/opt/kda_env/g2_c3; L=/public/home/tanbo/codex_validation/locks/bw46
cd /opt/kda_env && rm -rf g2_c3 && cp -a g2_r5 g2_c3 && cd g2_c3 && rm -rf build && python3 /public/home/tanbo/xplat/c3_patch.py . && python3 build_g2.py > /tmp/c3.log 2>&1 && echo build_ok || { grep -m5 error /tmp/c3.log; exit 1; }
exec 9>$L/hcu_5.lock; flock -w 900 9 || exit 75
export KDA_G2=1 HIP_VISIBLE_DEVICES=5 PYTHONPATH=/opt/kda_env:$T:/opt/kda_env/takeover
cd /tmp
G=$K/bitwise_gate.py
python3 $G dump --tree $T --T 8192 --H 12 --out /tmp/c3_prod.pt > /dev/null 2>&1
for n in 2 1; do KDA_G2_KPERM_MAX_H=0 KDA_G2_C3_NVH=$n python3 $G dump --tree $T --T 8192 --H 12 --out /tmp/c3_n$n.pt > /tmp/c3_n$n.log 2>&1 || tail -3 /tmp/c3_n$n.log
  echo "NVH=$n vs production: $(python3 $G cmp /tmp/c3_prod.pt /tmp/c3_n$n.pt | tail -1)"; done
for TH in "16384 12" "65536 12"; do set -- $TH
  echo "== T=$1 H=$2"
  echo "prod:  $(python3 $K/conc_probe.py --tree $T --T $1 --H $2 --reps 10 2>&1 | grep both_rev | grep -oE 'fwdh [0-9.]+ us')"
  for n in 2 1; do echo "NVH=$n: $(KDA_G2_KPERM_MAX_H=0 KDA_G2_C3_NVH=$n python3 $K/conc_probe.py --tree $T --T $1 --H $2 --reps 10 2>&1 | grep both_rev)"; done
done
echo C3_DONE
