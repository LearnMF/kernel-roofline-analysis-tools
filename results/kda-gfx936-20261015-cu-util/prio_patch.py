import sys
from pathlib import Path
f = Path(sys.argv[1]) / "csrc" / "g2" / "g2_fwd_v8.cuh"
s = f.read_text()
key = "g2_fwd_h_v8o4_kernel(FwdOParams p) {"
assert s.count(key) == 1
s = s.replace(key, key + "\n    __builtin_amdgcn_s_setprio(3);   // C-2 probe: chain waves first")
f.write_text(s)
print("PRIO APPLIED")
