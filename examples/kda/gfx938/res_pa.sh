#!/bin/bash
source /opt/dtk/env.sh >/dev/null 2>&1
for S in "$@"; do
  /opt/dtk/bin/hipcc -O3 -std=c++17 --offload-arch=gfx938 --cuda-device-only -S -DHIP_ENABLE_WARP_SYNC_BUILTINS -I$S/include -I$S/src/include -I$S/src/primitive/hygon/kda $S/src/primitive/hygon/kda/g2_kernels.hip -o /tmp/res_pa.s 2>/dev/null
  python3 - "$S" <<'PY'
import re, sys
txt = open("/tmp/res_pa.s").read()
for m in re.finditer(r"\.name:\s+(_ZN\S*g2_fwd_prep_a_kernel\S*)\n(.*?)(?=\n  - \.|\Z)", txt, re.S):
    n, body = m.group(1), m.group(2)
    g = lambda k: (re.search(rf"\.{k}:\s+(\d+)", body) or [None, "?"])[1]
    print(sys.argv[1].split("/")[-1], n[-40:], "vgpr", g("vgpr_count"), "lds", g("group_segment_fixed_size"), "spill", g("vgpr_spill_count"))
PY
done
