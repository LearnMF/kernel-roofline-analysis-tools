import json, re, sys, math
tab = open(sys.argv[1]).read().split("### Wall time")[0]
ref = {}
for l in tab.splitlines():
    m = re.match(r"\| (\d+)K \| TP(\d) \((\d+)\) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", l)
    if m:
        T, tp, H = int(m.group(1)) * 1024, int(m.group(2)), int(m.group(3))
        ref[(T, H)] = dict(tp=tp, p88=float(m.group(4)), a8d=float(m.group(7)), a8f=float(m.group(8)), h2d=float(m.group(9)), h2f=float(m.group(10)))
ft = json.load(open(sys.argv[2])); g2 = json.load(open(sys.argv[3]))
rows, gm = [], {k: [] for k in ("ftg2", "a8d", "a8f", "h2d", "h2f")}
print("| T | TP (H/card) | flash-train (chunk_kda) | hip_kda same session | ft / hip_kda | A800 FLA def | A800 FLA fast | H20 FLA def | H20 FLA fast | A800 def / ft | A800 fast / ft | H20 def / ft | H20 fast / ft |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for (T, H), r in sorted(ref.items()):
    k = f"{T}x{H}"
    f, h = ft[k]["ftc" if "ftc" in ft[k] else "g2"]["device_fb_ms"], g2[k]["g2"]["device_fb_ms"]
    v = dict(ftg2=f / h, a8d=r["a8d"] / f, a8f=r["a8f"] / f, h2d=r["h2d"] / f, h2f=r["h2f"] / f)
    for kk in gm: gm[kk].append(v[kk])
    b = lambda x: f"**{x:.2f}**" if x >= 1 else f"{x:.2f}"
    print(f"| {T//1024}K | TP{r['tp']} ({H}) | {f:.2f} | {h:.2f} | {v['ftg2']:.3f} | {r['a8d']:.2f} | {r['a8f']:.2f} | {r['h2d']:.2f} | {r['h2f']:.2f} | {b(v['a8d'])} | {b(v['a8f'])} | {b(v['h2d'])} | {b(v['h2f'])} |")
g = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))
print("\ngeomean (16 shapes): " + ", ".join(f"{k} {g(v):.3f} [{min(v):.3f}-{max(v):.3f}]" for k, v in gm.items()))
