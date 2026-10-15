"""Static per-kernel instruction categories from a production disassembly (extractkernel/llvm-objdump .isa).
Use for straight-line (unrolled) chunk-parallel kernels where static ~ dynamic.
    python3 isa_cat.py g2.isa <symbol-substring> [...]"""
import collections
import re
import sys

HDR = re.compile(r"^[0-9a-f]+ <(_Z\w+)>:")
OPC = re.compile(r"^\s+([vsd][a-z0-9_]+|global_\w+|flat_\w+|buffer_\w+)\b")


def cat(op):
    if op.startswith("v_mmac"):
        return "mmac"
    if op.startswith(("global_load", "buffer_load", "flat_load")):
        return "vmem_ld"
    if op.startswith(("global_store", "buffer_store", "flat_store")):
        return "vmem_st"
    if op.startswith("global_atomic"):
        return "vmem_atomic"
    if op.startswith("ds_bpermute") or op.startswith("ds_permute") or op.startswith("ds_swizzle"):
        return "lds_xlane"
    if op.startswith("ds_read"):
        return "lds_rd"
    if op.startswith("ds_write"):
        return "lds_wr"
    if op.startswith("s_waitcnt"):
        return "waitcnt"
    if op.startswith("s_"):
        return "salu"
    if op in ("v_add_co_u32_e32", "v_add_co_u32_e64", "v_addc_co_u32_e32", "v_addc_co_u32_e64",
              "v_sub_co_u32_e32", "v_sub_co_u32_e64", "v_subb_co_u32_e32", "v_subb_co_u32_e64",
              "v_lshlrev_b64", "v_lshrrev_b64", "v_mad_u64_u32", "v_mad_i64_i32", "v_ashrrev_i64"):
        return "valu_addr64"
    if op in ("v_bfe_u32", "v_add3_u32"):
        return "valu_bf16rnd"
    if op.startswith(("v_exp", "v_log", "v_rcp", "v_rsq", "v_sqrt", "v_sin", "v_cos")):
        return "valu_trans"
    if re.match(r"v_(pk_)?(fma|mul|add|sub|max|min|mac|mad)_f(32|16)|v_pk_(fma|mul|add)_f32|v_cvt|v_fmac|v_div|v_ldexp|v_frexp|v_cndmask", op):
        return "valu_fp"
    if op.startswith("v_mov") or op.startswith("v_readfirstlane") or op.startswith("v_readlane") or op.startswith("v_writelane"):
        return "valu_mov"
    if op.startswith("v_"):
        return "valu_int"
    return "other"


lines = open(sys.argv[1]).read().splitlines()
starts = [(i, HDR.match(l).group(1)) for i, l in enumerate(lines) if HDR.match(l)]
starts.append((len(lines), "END"))
for want in sys.argv[2:]:
    for j, (i, name) in enumerate(starts[:-1]):
        if want not in name:
            continue
        c = collections.Counter()
        ops = collections.Counter()
        for l in lines[i:starts[j + 1][0]]:
            m = OPC.match(l)
            if m:
                c[cat(m.group(1))] += 1
                ops[m.group(1)] += 1
        tot = sum(c.values())
        valu = sum(v for k, v in c.items() if k.startswith("valu"))
        print(f"\n{name}  total {tot}  VALU {valu}")
        print("  " + "  ".join(f"{k}={v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1])))
        print("  top VALU ops: " + ", ".join(f"{o}={v}" for o, v in ops.most_common(60) if o.startswith("v_") and not o.startswith("v_mmac"))[:900])
