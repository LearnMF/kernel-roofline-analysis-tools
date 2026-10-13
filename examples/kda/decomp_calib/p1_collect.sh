#!/bin/bash
# P1 residency sweep: run kda_iter on g2_calib with KDA_PROBE_LDS_<k>=KB per level,
# 5 jittered processes, median device time + one PMC pass per level.
#   bash p1_collect.sh <hcu> <T> <H>
# Env knobs (KB of total LDS per CTA; 0/empty = production residency):
#   PROBE_KERNELS="WU DAV TAIL S4A S4B PREPA PREPB FWH"   subset to sweep
#   PROBE_LEVELS_<k>="0 22 33 48"                        per-kernel LDS targets
set -euo pipefail
dev=$1 T=$2 H=$3 TREE=/opt/kda_env/g2_calib
L=/public/home/tanbo/codex_validation/locks
exec 8>"$L/bw7/hcu_${dev}.lock"; flock -w 600 8 || { echo LOCK_BUSY; exit 75; }
exec 9>"$L/hcu_${dev}.lock";     flock -w 600 9 || { echo LOCK_BUSY; exit 75; }
docker exec -e HIP_VISIBLE_DEVICES=$dev -e KDA_G2=1 -e KDA_SKIP_FLA_PRUNE=1 -e OTAG=${T}_${H} \
  -e PYTHONPATH=/opt/kda_env:/public/home/tanbo/xplat/bwenv:$TREE:/public/home/tanbo/codex_validation/stage_g2_bw7/takeover \
  -e PROBE_KERNELS="${PROBE_KERNELS:-}" -e PROBE_SPEC="${PROBE_SPEC:-}" \
  tanbo_mega_k3 bash -lc '
export PATH=/opt/dtk/bin:$PATH
O=/public/home/tanbo/kra_cards/p1_${OTAG}; mkdir -p $O
K="python3 /public/home/tanbo/hier_pmc/kda_run.py --tree '"$TREE"' --arm g2 --T '"$T"' --H '"$H"'"
nonempty(){ [ -s "$1" ]; }
# spec lines: "<tag> <env assignments...>"  e.g. "wu_22 KDA_PROBE_LDS_WU=22"
SPEC="$PROBE_SPEC"
if [ -z "$SPEC" ]; then
  SPEC="base "
  for k in $PROBE_KERNELS; do
    v=PROBE_LEVELS_$k; eval "lv=\"\${$v:-22 33}\""
    for kb in $lv; do SPEC="$SPEC
${k}_${kb} KDA_PROBE_LDS_${k}=${kb}"; done
  done
fi
echo "$SPEC" | while read -r tag envs; do
  [ -z "$tag" ] && continue
  ok=1
  for i in 1 2 3 4 5; do nonempty "$O/${tag}_trace_$i.json" || ok=0; done
  nonempty "$O/${tag}_pmc_read.csv" || ok=0
  [ $ok -eq 1 ] && { echo "skip $tag"; continue; }
  for i in 1 2 3 4 5; do
    nonempty "$O/${tag}_trace_$i.json" || { rm -f "$O/${tag}_trace_$i.json" "$O/${tag}_trace_$i.db"; \
      KRA_JITTER_SEED=$i timeout 1200 env $envs hipprof --hip-trace --output-type 0 -o "$O/${tag}_trace_$i" $K --iters 4 > "$O/${tag}_trace_$i.log" 2>&1; }
  done
  nonempty "$O/${tag}_pmc_read.csv"  || timeout 1200 env $envs hipprof --pmc-read  --pmc-type 3 -o "$O/${tag}_pmc_read"  $K --iters 1 > "$O/${tag}_pmc_read.log"  2>&1
  nonempty "$O/${tag}_pmc_write.csv" || timeout 1200 env $envs hipprof --pmc-write --pmc-type 3 -o "$O/${tag}_pmc_write" $K --iters 1 > "$O/${tag}_pmc_write.log" 2>&1
  echo "done $tag"
done
echo P1_DONE_${OTAG}
'
