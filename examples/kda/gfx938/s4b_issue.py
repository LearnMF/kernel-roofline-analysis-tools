"""Issue-time attribution of one non-recurrent kernel dispatch from hipprof SQTT (--sqtt-type 1, json per SE).
Per wave: instruction i costs its own clks (issue) plus the gap since the previous instruction's issue end (stall),
both charged to i's class.  Averages over all traced waves; top PCs by stall.
    python3 s4b_issue.py <dir> [dispatch-index]"""
import collections, glob, json, re, sys
d = sys.argv[1]; disp = sys.argv[2] if len(sys.argv) > 2 else "0"
def cls(a):
    op = a.split()[0]
    if op.startswith("v_mmac") and "16x16x8_f32" in op: return "mmac_f32x8"
    if op.startswith("v_mmac"): return "mmac_bf16"
    if op == "ds_bpermute_b32": return "lds_bpermute"
    if op.startswith("ds_read") or op.startswith("ds_swizzle"): return "lds_read"
    if op.startswith("ds_write"): return "lds_write"
    if op.startswith("ds_"): return "lds_other"
    if op.startswith(("global_load", "buffer_load", "flat_load")): return "vmem_load"
    if op.startswith(("global_store", "buffer_store", "flat_store", "global_atomic")): return "vmem_store"
    if op.startswith("s_waitcnt"):
        return "wait_vm" if "vmcnt" in a else ("wait_lgkm" if "lgkmcnt" in a else "wait_other")
    if op.startswith("s_barrier"): return "barrier"
    if op.startswith(("s_load", "s_buffer_load")): return "smem"
    if op.startswith(("v_exp", "v_log", "v_rcp", "v_rsq", "v_sqrt")): return "valu_trans"
    if re.match(r"v_(bfe|add3|lshr|lshl|and|or|xor|perm|alignbit|cndmask|mov|add_u32|add_co|addc|sub_u32|mad_u|mul_lo|mul_hi|lshl_add|lshl_or|and_or|bfi|readfirstlane|readlane|writelane|cvt_pk_bf16|cvt_pk)", op): return "valu_int_move"
    if op.startswith("v_"): return "valu_fp"
    return "salu"
issue = collections.Counter(); stall = collections.Counter(); cnt = collections.Counter()
pc_stall = collections.Counter(); pc_asm = {}; spans = []; nw = 0
for f in sorted(glob.glob(f"{d}/*thread_trace_{disp}__*.json")):
    ev = json.load(open(f))["traceEvents"]
    waves = collections.defaultdict(list)
    for e in ev:
        a = e.get("args", {})
        if e.get("ph") == "X" and "assembly" in a: waves[(e["pid"], e["tid"])].append((a["time"], a.get("clks", 4), a["assembly"].strip(), a.get("offset")))
    for w, xs in waves.items():
        xs.sort(); nw += 1; spans.append(xs[-1][0] - xs[0][0])
        prev_end = xs[0][0]
        for t, c, asm, off in xs:
            k = cls(asm); g = max(0, t - prev_end)
            issue[k] += c; stall[k] += g; cnt[k] += 1
            pc_stall[off] += g; pc_asm[off] = asm
            prev_end = t + c
tot = sum(issue.values()) + sum(stall.values())
print(f"dispatch {disp}: {nw} waves, mean wave span {sum(spans)/len(spans):.0f} cycles, accounted {tot/nw:.0f} cycles/wave")
print(f"{'class':16s} {'instrs/wave':>11s} {'issue cyc':>10s} {'stall cyc':>10s} {'total %':>8s}")
for k in sorted(issue, key=lambda k: -(issue[k] + stall[k])):
    print(f"{k:16s} {cnt[k]/nw:11.1f} {issue[k]/nw:10.0f} {stall[k]/nw:10.0f} {100*(issue[k]+stall[k])/tot:7.1f}%")
print("top stall PCs (cycles/wave):")
for off, s in pc_stall.most_common(25):
    print(f"  {off:>8s} {s/nw:8.1f}  {pc_asm[off][:90]}")
