#!/bin/bash
# usage: insn_probe.sh ARCH SRCDIR  -> builtin availability, FLA Triton cache instruction counts, flash-train ISA counts
ARCH=$1; S=$2
source /opt/dtk/env.sh >/dev/null 2>&1
cat > /tmp/hb.hip <<'H'
#include <hip/hip_runtime.h>
#if __has_builtin(__builtin_hcu_ds_read_m32x16_bf16)
HAS___ds_read_m32x16_bf16
#endif
#if __has_builtin(__builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts)
HAS___mmac_lit_lts
#endif
#if __has_builtin(__builtin_hcu_cvt_pk_bf16_f32)
HAS___cvt_pk_bf16_f32
#endif
#if __has_builtin(__builtin_hcu_matrix_load_32x32_b16)
HAS___matrix_load_32x32_b16
#endif
#if __has_builtin(__builtin_hcu_ds_read_matrix_format)
HAS___ds_read_matrix_format
#endif
H
echo "== builtins visible for $ARCH:"; /opt/dtk/bin/hipcc --offload-arch=$ARCH --cuda-device-only -E /tmp/hb.hip 2>/dev/null | grep -o "HAS___[a-z0-9_]*" | sort -u
echo "== FLA Triton cache ($ARCH), KDA-related kernels:"
python3 - <<'PY'
import glob, re, collections, os
files = glob.glob(os.path.expanduser("~/.triton/cache/*/*.amdgcn"))
pat = {"ds_read_m32x16": r"ds_read_m32x16", "ds_read_matrix": r"ds_read_matrix", "mmac_any": r"v_mmac_\w+", "mmac_lit": r"v_mmac_\w*lit\w*",
       "cvt_pk_bf16": r"v_cvt_pk_bf16", "matrix_load": r"matrix_load", "ds_bpermute": r"ds_bpermute", "ds_write": r"ds_write\w*"}
tot = collections.Counter(); n = 0; mm = collections.Counter(); arch = collections.Counter()
for f in files:
    s = open(f, errors="ignore").read()
    name = re.search(r"\.amdhsa_kernel (\S+)", s)
    a = re.search(r"amdgcn-amd-amdhsa--(gfx\w+)", s); arch[a.group(1) if a else "?"] += 1
    if not name or not re.search(r"kda|wy|dqkg|gla|delta|recompute|intra|chunk", name.group(1)): continue
    n += 1
    for k, p in pat.items(): tot[k] += len(re.findall(r"^\s+" + p, s, re.M))
    for m in re.findall(r"^\s+(v_mmac_\w+)", s, re.M): mm[m] += 1
print(f"  cache archs {dict(arch)}; {n} KDA-related Triton kernels:", dict(tot))
print("  MMAC opcodes in them:", mm.most_common(8))
PY
echo "== flash-train hand-written KDA kernels ($ARCH):"
/opt/dtk/bin/hipcc -O3 -std=c++17 --offload-arch=$ARCH --cuda-device-only -S -DHIP_ENABLE_WARP_SYNC_BUILTINS -I$S/include -I$S/src/include -I$S/src/primitive/hygon/kda $S/src/primitive/hygon/kda/g2_kernels.hip -o /tmp/ft_$ARCH.s 2>/dev/null
python3 - <<PY
import re, collections
s = open("/tmp/ft_$ARCH.s").read()
pat = {"ds_read_m32x16": r"ds_read_m32x16", "ds_read_matrix": r"ds_read_matrix", "mmac_any": r"v_mmac_\w+", "mmac_lit": r"v_mmac_\w*lit\w*",
       "cvt_pk_bf16": r"v_cvt_pk_bf16", "matrix_load": r"matrix_load", "ds_bpermute": r"ds_bpermute", "ds_write": r"ds_write\w*"}
print("  ", {k: len(re.findall(r"^\s+" + p, s, re.M)) for k, p in pat.items()})
print("   MMAC opcodes:", collections.Counter(re.findall(r"^\s+(v_mmac_\w+)", s, re.M)).most_common(6))
PY
