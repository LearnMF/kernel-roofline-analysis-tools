import collections, glob, json, sys
d = sys.argv[1]; disp = sys.argv[2] if len(sys.argv) > 2 else "0"
op_cnt = collections.Counter(); op_stall = collections.Counter(); pc = collections.Counter(); pcs = collections.Counter(); asmof = {}; nw = 0
wait_after = collections.Counter()
for f in sorted(glob.glob(f"{d}/*thread_trace_{disp}__*.json")):
    ev = json.load(open(f))["traceEvents"]
    waves = collections.defaultdict(list)
    for e in ev:
        a = e.get("args", {})
        if e.get("ph") == "X" and "assembly" in a: waves[(e["pid"], e["tid"])].append((a["time"], a.get("clks", 4), a["assembly"].strip(), a.get("offset")))
    for w, xs in waves.items():
        xs.sort(); nw += 1; pe = xs[0][0]
        for t, c, asm, off in xs:
            g = max(0, t - pe); pe = t + c; op = asm.split()[0]
            if op.startswith(("global_", "buffer_")):
                op_cnt[op] += 1; op_stall[op] += g; pc[off] += 1; pcs[off] += g; asmof[off] = asm
print("VMEM by opcode (per wave): count, stall cycles before issue")
for op in sorted(op_cnt, key=lambda o: -op_stall[o]): print(f"  {op:28s} {op_cnt[op]/nw:7.1f} {op_stall[op]/nw:9.0f}")
print("distinct VMEM PCs:", len(pc))
offs = sorted(pc, key=lambda o: int(o, 16))
# group into runs of PCs within 0x200 bytes
runs = []; cur = [offs[0]]
for o in offs[1:]:
    if int(o, 16) - int(cur[-1], 16) < 0x200: cur.append(o)
    else: runs.append(cur); cur = [o]
runs.append(cur)
print("VMEM PC regions (start, n instrs, executions/wave, stall/wave, sample):")
for r in sorted(runs, key=lambda r: -sum(pcs[o] for o in r))[:14]:
    print(f"  {r[0]:>8s} n={len(r):3d} exec={sum(pc[o] for o in r)/nw:6.1f} stall={sum(pcs[o] for o in r)/nw:7.0f}  {asmof[r[0]][:70]}")
