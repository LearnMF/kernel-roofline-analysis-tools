#!/bin/bash
# ISA opcode counts of the gfx938 production prep_a (GATE=1, VL=0, L2N=0) for two source trees
source /opt/dtk/env.sh >/dev/null 2>&1
for S in "$@"; do
  /opt/dtk/bin/hipcc -O3 -std=c++17 --offload-arch=gfx938 --cuda-device-only -S -DHIP_ENABLE_WARP_SYNC_BUILTINS -I$S/include -I$S/src/include -I$S/src/primitive/hygon/kda $S/src/primitive/hygon/kda/g2_kernels.hip -o /tmp/isa_pa.s 2>/dev/null
  python3 - <<'PY'
import re, collections
cur = None; c = collections.Counter(); meta = {}
for l in open("/tmp/isa_pa.s"):
    m = re.match(r"^(_ZN[\w$]+):", l)
    if m: cur = m.group(1) if ("g2_fwd_prep_a_kernel" in m.group(1) and "ILi1ELb0ELb0E" in m.group(1)) else None; continue
    if cur and re.match(r"^\s*\.Lfunc_end", l): cur = None; continue
    if cur and re.match(r"^\s+[sv]_|^\s+ds_|^\s+global_", l): c[l.split()[0]] += 1
    if re.match(r"^\s+\.(vgpr_count|sgpr_count|group_segment_fixed_size):", l) and "prep_a" in str(meta.get("k", "")): pass
print("total", sum(c.values()), " ".join(f"{k}={c[k]}" for k in ["v_sub_f32_e32", "v_pk_add_f32", "v_add_f32_e32", "v_pk_mul_f32", "v_mul_f32_e32", "v_exp_f32_e32", "v_mov_b32_e32", "v_mov_b64_e32"]))
PY
done
