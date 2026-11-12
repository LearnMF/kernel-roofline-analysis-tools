"""Top opcodes (dynamic count per wave, issue+stall cycles) of one kernel's SQTT dir (last dispatch, MAXF SE files)."""
import collections, glob, json, os, re, sys
d = sys.argv[1]; maxf = int(sys.argv[2]) if len(sys.argv) > 2 else 2
files = sorted(glob.glob(f"{d}/*thread_trace_*__*.json"))
disp = sorted({re.search(r"thread_trace_(\d+)__", f).group(1) for f in files}, key=int)[-1]
cnt, cyc = collections.Counter(), collections.Counter(); nw = 0
for f in [f for f in files if f"thread_trace_{disp}__" in f][:maxf]:
    ev = json.load(open(f))["traceEvents"]; waves = collections.defaultdict(list)
    for e in ev:
        a = e.get("args", {})
        if e.get("ph") == "X" and "assembly" in a: waves[(e["pid"], e["tid"])].append((a["time"], a.get("clks", 4), a["assembly"].strip()))
    for xs in waves.values():
        xs.sort(); nw += 1; prev = xs[0][0]
        for t, c, asm in xs:
            op = asm.split()[0]; cnt[op] += 1; cyc[op] += c + max(0, t - prev); prev = t + c
tc = sum(cyc.values())
print(f"{d}: {nw} waves")
for op, n in cnt.most_common(40): print(f"  {op:32s} {n/nw:8.1f}/wave  cycles {100*cyc[op]/tc:5.1f}%")
