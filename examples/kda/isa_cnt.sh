#!/bin/bash
# isa_cnt.sh TREE OUT.s : per-instance instruction counts of g2_fwd_h_v8o4_kernel (dense) and prep_b
[ -n "$NOBUILD" ] || { cd $1 && bash isa_g2.sh $2 2>/dev/null || exit 1; }
for k in $(grep -oE "^_ZN2g220g2_fwd_h_v8o4_kernelILb0ELi[0-9]+E[A-Za-z0-9_]*:" $2 | tr -d : | sort -u) $(grep -oE "^_ZN2g22r420g2_fwd_prep_b_kernelILb0EEEvNS0_11PrepBParamsE:" $2 | tr -d :); do
  awk -v k="$k" '$0 ~ "^"k":" {f=1; next} f && /s_endpgm/ {exit} f' $2 > /tmp/isa_k.s
  printf "%-60s tot=%5d dsw16=%3d dsw64=%3d dsr16=%3d mmac=%3d vmem=%3d bperm=%3d valu=%5d salu=%4d\n" "$k" \
    $(grep -cE "^[[:space:]]+[a-z]" /tmp/isa_k.s) $(grep -cE "ds_write_b16|ds_write_short" /tmp/isa_k.s) $(grep -cE "ds_write_b64" /tmp/isa_k.s) \
    $(grep -cE "ds_read_u16|ds_read_i16" /tmp/isa_k.s) $(grep -cE "v_mmac" /tmp/isa_k.s) $(grep -cE "global_|buffer_" /tmp/isa_k.s) \
    $(grep -cE "ds_bpermute" /tmp/isa_k.s) $(grep -cE "^[[:space:]]+v_" /tmp/isa_k.s) $(grep -cE "^[[:space:]]+s_" /tmp/isa_k.s)
done
