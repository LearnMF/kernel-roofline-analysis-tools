#!/bin/bash
# isa_k.sh SRC KEY... : static opcode histogram + resources of every gfx938 kernel whose symbol contains KEY
source /opt/dtk/env.sh >/dev/null 2>&1; S=$1; shift
/opt/dtk/bin/hipcc -O3 -std=c++17 --offload-arch=gfx938 --cuda-device-only -S -DHIP_ENABLE_WARP_SYNC_BUILTINS -I$S/include -I$S/src/include -I$S/src/primitive/hygon/kda $S/src/primitive/hygon/kda/g2_kernels.hip -o /tmp/isa_k.s 2>/dev/null
python3 - "$@" <<'PY'
import re, sys, collections
txt = open("/tmp/isa_k.s").read().split("\n")
funcs = {}; cur = None
for l in txt:
    m = re.match(r"^(_ZN[\w$]+):", l)
    if m: cur = m.group(1); funcs[cur] = collections.Counter(); continue
    if cur and re.match(r"^\s*\.Lfunc_end", l): cur = None; continue
    if cur and re.match(r"^\s+[sv]_|^\s+ds_|^\s+global_|^\s+buffer_", l): funcs[cur][l.split()[0]] += 1
meta = "\n".join(txt)
for key in sys.argv[1:]:
    for n, c in funcs.items():
        if key not in n: continue
        mm = re.search(rf"\.name:\s+{re.escape(n)}\n(.*?)(?=\n  - \.|\Z)", meta, re.S)
        g = lambda k: (re.search(rf"\.{k}:\s+(\d+)", mm.group(1)) or [None, "?"])[1] if mm else "?"
        print(f"{n[:70]}  total {sum(c.values())}  vgpr {g('vgpr_count')}  lds {g('group_segment_fixed_size')}")
        print("   " + "  ".join(f"{k}={v}" for k, v in c.most_common(28)))
PY
