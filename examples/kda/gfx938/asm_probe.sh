#!/bin/bash
# Assemble each instruction for gfx936 and gfx938: does the hardware ISA (assembler) accept it?
source /opt/dtk/env.sh >/dev/null 2>&1
MC=$(ls /opt/dtk/llvm/bin/llvm-mc /opt/dtk/bin/llvm-mc /opt/dtk/lib/llvm/bin/llvm-mc 2>/dev/null | head -1)
[ -z "$MC" ] && MC=$(find /opt/dtk -name llvm-mc -type f 2>/dev/null | head -1)
echo "llvm-mc: $MC"
while IFS= read -r ins; do
  for a in gfx936 gfx938; do
    out=$(echo "$ins" | $MC -arch=amdgcn -mcpu=$a -show-encoding 2>&1)
    if echo "$out" | grep -q "error"; then r="REJECT: $(echo "$out" | grep -m1 error | sed 's/.*error: //' | cut -c1-60)"; else r="ok $(echo "$out" | grep -o 'encoding: \[[^]]*\]' | cut -c1-48)"; fi
    printf "%-7s %-70s %s\n" "$a" "$ins" "$r"
  done
done <<'L'
ds_read_m32x16_b16 v[0:3], v4
v_mmac_f32_16x16x16_bf16 v[0:3], v[4:5], v[6:7], v[0:3]
v_mmac_f32_16x16x16_bf16 v[0:3], v[4:5], v[6:7], v[0:3] lit
v_cvt_pk_bf16_f32 v9, v1, v1
matrix_load_32x32_b16 s[0:3] s25 lds
L
