#!/bin/bash
# P2 reg-pressure loop: args like "s4b:0 s4b:8 s4b:16 ..."
# For each: restore pristine g2_wu.cuh+g2_wy.cuh from g2_calib, inject N pins,
# rebuild g2_calib_p2, run 4 iters on the two 16K shapes, log VGPR/scratch.
TREE=/opt/kda_env/g2_calib_p2
O=/public/home/tanbo/kra_cards/p2; mkdir -p "$O"
export PATH=/opt/dtk/bin:$PATH
for spec in "$@"; do
  IFS=: read -r tag N <<<"$spec"
  case $tag in
    wu)  hdr=g2_wu.cuh; kern=g2_wu_kernel;;
    s4a) hdr=g2_wy.cuh; kern=p3_wy_s4a_kernel;;
    s4b) hdr=g2_wy.cuh; kern=p3_wy_s4b_kernel;;
    *) echo "bad tag $tag"; continue;;
  esac
  for h in g2_wu.cuh g2_wy.cuh; do cp /opt/kda_env/g2_calib/csrc/g2/$h $TREE/csrc/g2/$h; done
  python3 /public/home/tanbo/kra/examples/kda/decomp_calib/patch_reg_inject.py $TREE $hdr $kern $N
  (cd $TREE && python3 build_g2.py) > "$O/${tag}_n${N}_build.log" 2>&1 \
    || { echo "BUILD_FAIL_$spec"; continue; }
  for TH in "16384 48" "16384 12"; do
    set -- $TH; T=$1; H=$2
    hipprof --hip-trace --output-type 0 -o "$O/${tag}_n${N}_${T}_${H}_t" \
      python3 /public/home/tanbo/hier_pmc/kda_run.py --tree $TREE --arm g2 \
        --T $T --H $H --iters 4 > "$O/${tag}_n${N}_${T}_${H}_t.log" 2>&1
  done
  echo "DONE_$spec"
done
echo P2_ALL_DONE
