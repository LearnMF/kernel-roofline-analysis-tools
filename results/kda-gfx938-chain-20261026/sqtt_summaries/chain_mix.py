"""Whole-chain dynamic instruction budget on gfx938 from SQTT (one directory per kernel, hipprof --sqtt-type 1).

For every kernel: per-wave dynamic instruction counts, issue cycles and stall cycles by class, where VALU is split
into real math (fp, transcendental) and data-movement work (format conversion / packing, integer / address,
logic / shifts, select / move / compare).  Each kernel's last traced dispatch is used.  Shares are weighted by the
kernel's measured time when a times JSON ({kernel_dir_name: us}) is given.
    python3 chain_mix.py <chain_dir> [times.json]"""
import collections
import glob
import json
import os
import re
import sys

FP = re.compile(r"v_(pk_)?(fma|fmac|mul|add|sub|subrev|max|min|mad|med3|max3|min3|ldexp|frexp|fract|floor|ceil|"
                r"rndne|trunc|div_|dot)\w*_f(16|32|64)")
TRANS = re.compile(r"v_(exp|log|rcp|rsq|sqrt|sin|cos)_")
CVT = re.compile(r"v_(cvt_|perm_b32|pack_|alignbit|alignbyte|bfe_|bfi_|lshl_or_b32|and_or_b32|or3_b32)")
INT = re.compile(r"v_(add_u32|add_co_u32|addc_co_u32|add_nc_u32|sub_u32|subrev_u32|sub_co_u32|subrev_co_u32|"
                 r"subb_co_u32|mad_u32_u24|mad_i32_i24|mad_u64_u32|mul_lo_u32|mul_hi_u32|mul_u32_u24|mul_i32_i24|"
                 r"lshl_add_u32|add3_u32|add_lshl_u32|mad_u32_u16|ashrrev|lshlrev_b64|lshrrev_b64)")
LOGIC = re.compile(r"v_(and_b32|or_b32|xor_b32|not_b32|lshlrev_b32|lshrrev_b32|lshlrev_b16|lshrrev_b16|bcnt|ffbh|ffbl)")
SEL = re.compile(r"v_(cndmask|mov_|readfirstlane|readlane|writelane|cmp|cmpx|swap)")


def cls(asm):
    op = asm.split()[0]
    if op.startswith("v_mmac"):
        return "mmac"
    if "matrix_load" in op:
        return "mls"
    if op.startswith(("global_load", "buffer_load", "flat_load")):
        return "vmem_ld"
    if op.startswith(("global_store", "buffer_store", "flat_store", "global_atomic")):
        return "vmem_st"
    if op.startswith(("ds_bpermute", "ds_permute", "ds_swizzle")):
        return "lds_xlane"
    if op.startswith("ds_read_m") or "matrix" in op:
        return "lds_rd_matrix"
    if op.startswith("ds_read"):
        return "lds_rd"
    if op.startswith("ds_write"):
        return "lds_wr"
    if op.startswith("ds_"):
        return "lds_other"
    if op.startswith("s_waitcnt"):
        return "wait_vm" if "vmcnt" in asm else ("wait_lgkm" if "lgkmcnt" in asm else "wait_other")
    if op.startswith("s_barrier"):
        return "barrier"
    if op.startswith(("s_load", "s_buffer_load")):
        return "smem"
    if op.startswith("v_"):
        if TRANS.match(op):
            return "v_trans"
        if FP.match(op):
            return "v_fp"
        if CVT.match(op):
            return "v_cvt_pack"
        if INT.match(op):
            return "v_int_addr"
        if LOGIC.match(op):
            return "v_logic"
        if SEL.match(op):
            return "v_sel_move"
        return "v_other"
    return "salu"


GROUPS = {
    "compute": ("mmac", "v_fp", "v_trans"),
    "valu_move": ("v_cvt_pack", "v_int_addr", "v_logic", "v_sel_move", "v_other"),
    "lds": ("lds_rd", "lds_wr", "lds_xlane", "lds_rd_matrix", "lds_other"),
    "vmem": ("vmem_ld", "vmem_st", "mls", "smem"),
    "sync": ("barrier", "wait_vm", "wait_lgkm", "wait_other"),
    "salu": ("salu",),
}


def analyse(kdir):
    files = sorted(glob.glob(f"{kdir}/*thread_trace_*__*.json"))
    if not files:
        return None
    disp = sorted({re.search(r"thread_trace_(\d+)__", f).group(1) for f in files}, key=int)[-1]
    issue, stall, cnt = collections.Counter(), collections.Counter(), collections.Counter()
    nw, spans = 0, []
    sel = [f for f in files if f"thread_trace_{disp}__" in f][:int(os.environ.get("MAXF", "999"))]   # SE sample
    for f in sel:
        ev = json.load(open(f))["traceEvents"]
        waves = collections.defaultdict(list)
        for e in ev:
            a = e.get("args", {})
            if e.get("ph") == "X" and "assembly" in a:
                waves[(e["pid"], e["tid"])].append((a["time"], a.get("clks", 4), a["assembly"].strip()))
        for xs in waves.values():
            xs.sort()
            nw += 1
            spans.append(xs[-1][0] - xs[0][0])
            prev = xs[0][0]
            for t, c, asm in xs:
                k = cls(asm)
                issue[k] += c
                stall[k] += max(0, t - prev)
                cnt[k] += 1
                prev = t + c
    if not nw:
        return None
    tot = sum(issue.values()) + sum(stall.values())
    return dict(disp=disp, waves=nw, span=sum(spans) / nw, tot=tot / nw,
                cnt={k: v / nw for k, v in cnt.items()}, issue={k: v / nw for k, v in issue.items()},
                stall={k: v / nw for k, v in stall.items()})


def main():
    root = sys.argv[1]
    times = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
    res = {}
    for kd in sorted(os.listdir(root)):
        p = os.path.join(root, kd)
        if os.path.isdir(p):
            r = analyse(p)
            if r:
                res[kd] = r
    print(f"{'kernel':22s} {'waves':>5s} {'cyc/wave':>9s} | instr/wave: " + " ".join(f"{g:>9s}" for g in GROUPS) +
          " | VALU split: fp trans | cvt int logic sel | move/(fp+trans)")
    agg_cnt, agg_cyc, wsum = collections.Counter(), collections.Counter(), 0.0
    for kd, r in res.items():
        gc = {g: sum(r["cnt"].get(k, 0) for k in ks) for g, ks in GROUPS.items()}
        c = r["cnt"]
        math_ = c.get("v_fp", 0) + c.get("v_trans", 0)
        move = sum(c.get(k, 0) for k in GROUPS["valu_move"])
        print(f"{kd:22s} {r['waves']:5d} {r['tot']:9.0f} | " + " ".join(f"{gc[g]:9.0f}" for g in GROUPS) +
              f" | {c.get('v_fp', 0):6.0f} {c.get('v_trans', 0):5.0f} | {c.get('v_cvt_pack', 0):5.0f} "
              f"{c.get('v_int_addr', 0):5.0f} {c.get('v_logic', 0):5.0f} {c.get('v_sel_move', 0):5.0f} | "
              f"{move / max(math_, 1):5.2f}")
        cyc = {k: r["issue"].get(k, 0) + r["stall"].get(k, 0) for k in set(r["issue"]) | set(r["stall"])}
        print(f"{'':22s} cycles %: " + "  ".join(f"{g} {100 * sum(cyc.get(k, 0) for k in ks) / r['tot']:4.1f}" for g, ks in GROUPS.items())
              + f"  | wait_vm {100 * cyc.get('wait_vm', 0) / r['tot']:4.1f} barrier {100 * cyc.get('barrier', 0) / r['tot']:4.1f}")
        w = times.get(kd, 1.0)
        wsum += w
        tc = sum(r["cnt"].values())
        tcy = r["tot"]
        for k, v in r["cnt"].items():
            agg_cnt[k] += w * v / tc
        for k in set(r["issue"]) | set(r["stall"]):
            agg_cyc[k] += w * (r["issue"].get(k, 0) + r["stall"].get(k, 0)) / tcy
    print("\ntime-weighted chain shares (weights = kernel times):")
    print(f"{'class':14s} {'instr %':>8s} {'cycles %':>9s}")
    for k in sorted(agg_cyc, key=lambda k: -agg_cyc[k]):
        print(f"{k:14s} {100 * agg_cnt[k] / wsum:7.1f}% {100 * agg_cyc[k] / wsum:8.1f}%")
    for g, ks in GROUPS.items():
        print(f"  group {g:10s} instr {100 * sum(agg_cnt[k] for k in ks) / wsum:5.1f}%  cycles "
              f"{100 * sum(agg_cyc[k] for k in ks) / wsum:5.1f}%")
    json.dump(res, open(os.path.basename(os.path.normpath(root)) + "_mix.json", "w"), indent=1)


if __name__ == "__main__":
    main()
