#!/usr/bin/env bash
set -u
L=/public/home/tanbo/codex_validation/locks
exec 8>"$L/bw7/hcu_0.lock"; flock -w 300 8 || exit 75
exec 9>"$L/hcu_0.lock"; flock -w 300 9 || exit 75
docker exec \
  -e HIP_VISIBLE_DEVICES=0 -e KDA_SKIP_FLA_PRUNE=1 -e KDA_G2=1 \
  -e PYTHONPATH=/public/home/tanbo/xplat/bwenv:/opt/kda_env/g2_r5:/public/home/tanbo/codex_validation/stage_g2_bw7/takeover \
  tanbo_mega_k3 bash -lc '
for m in torch cuda_alloc fla triton fla.ops.kda fla_prune_ns4 hip_kda g2_ops fla_chunk g2_chunk; do
  echo "== $m"
  timeout 180 python3 /public/home/tanbo/hier_pmc/imp_probe.py "$m" 2>&1 | grep -v "Python version" | tail -3
done'
