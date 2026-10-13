#!/bin/bash
# P2 register metadata: for each (tag,N) re-inject into a scratch copy of the
# header and compile to .s (no .so build) -> .vgpr_count / spill counts.
# Args: "s4b:0 s4b:8 ... wu:16 ..."
set -e
export PATH=/opt/dtk/bin:$PATH
SRC=/opt/kda_env/g2_calib/csrc/g2          # launcher-probe tree == prod kernel code
W=/tmp/p2regs; rm -rf $W; mkdir -p $W
cp $SRC/g2_wu.cuh $SRC/g2_wy.cuh $W/
cp $SRC/g2_common.h $SRC/g2_config.h $W/ 2>/dev/null || true
O=/public/home/tanbo/kra_cards/p2; mkdir -p "$O"
for spec in "$@"; do
  IFS=: read -r tag N <<<"$spec"
  case $tag in
    wu)  hdr=g2_wu.cuh; kern=g2_wu_kernel;;
    s4a) hdr=g2_wy.cuh; kern=p3_wy_s4a_kernel;;
    s4b) hdr=g2_wy.cuh; kern=p3_wy_s4b_kernel;;
  esac
  cp $SRC/$hdr $W/$hdr
  # inject into the scratch copy; patch script writes to <tree>/csrc/g2/<hdr>
  mkdir -p $W/tree/csrc/g2 && cp $W/$hdr $W/tree/csrc/g2/$hdr
  python3 /public/home/tanbo/kra/examples/kda/decomp_calib/patch_reg_inject.py \
      $W/tree $hdr $kern $N > /dev/null
  hipcc -S -O3 -std=c++17 -I$W -I/opt/kda_env/g2_calib/csrc/g2 -I/opt/kda_env/g2_calib/csrc \
      --offload-arch=gfx936 -o $W/${tag}_n${N}.s -x hip $W/tree/csrc/g2/$hdr 2>/dev/null \
    || { echo "S_FAIL_$spec"; continue; }
  python3 - <<PY > "$O/${tag}_n${N}_regs.txt"
import re
s=open("$W/${tag}_n${N}.s").read()
for m in re.finditer(r"\.amdhsa_kernel\s+(\S+)(.*?)\.end_amdhsa_kernel", s, re.S):
    name=m.group(1); body=m.group(2)
    if "$kern".split("(")[0].replace("g2_","").replace("p3_","") in name or "$kern" in name:
        for f in ("vgpr_count","sgpr_count","vgpr_spill_count","sgpr_spill_count","group_segment_fixed_size","private_segment_fixed_size","wavefront_size","kernarg_segment_size"):
            mm=re.search(rf"\.{f}:\s*(\d+)", body)
            if mm: print(f, mm.group(1))
        print("KERNEL", name)
PY
  echo "REGS_${tag}_${N}"
done
