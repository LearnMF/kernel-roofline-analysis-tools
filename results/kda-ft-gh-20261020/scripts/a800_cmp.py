"""A800 FLA re-measure (2026-10-20) vs 10-17, and vs flash-train / hip_kda (BW device time from the jittered round,
wall from the same-process op bench).  python3 a800_cmp.py new.jsonl old.jsonl round_ft.json round_hip.json wall.jsonl"""
import json, sys, math, statistics as st, collections
def nv(path):
    d = collections.defaultdict(lambda: collections.defaultdict(list))
    for l in open(path):
        r = json.loads(l)["data"]
        for a in ("fla", "fla_nr"):
            for m in ("kernel_fb_ms", "fb_ms"): d[(r["T"], r["H"])][(a, m)].append(r["res"][a][m])
    return {k: {kk: st.median(v) for kk, v in vv.items()} for k, vv in d.items()}
new, old = nv(sys.argv[1]), nv(sys.argv[2])
ft, hip = json.load(open(sys.argv[3])), json.load(open(sys.argv[4]))
wall = collections.defaultdict(lambda: collections.defaultdict(list))
for l in open(sys.argv[5]):
    r = json.loads(l)["data"]
    for a in ("g2", "ftc"): wall[(r["T"], r["H"])][a].append(r["res"][a]["fb_ms"])
tp = {12: 8, 24: 4, 48: 2, 96: 1}
gm = collections.defaultdict(list)
print("### Device time (ms, one fwd+bwd); ratios A800 / ours (> 1 = ours faster)\n")
print("| T | TP (H) | A800 FLA def 10-20 | (10-17) | A800 FLA fast 10-20 | (10-17) | flash-train | hip_kda | def / ft | fast / ft | def / hip | fast / hip |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for (T, H) in sorted(new):
    n, o = new[(T, H)], old[(T, H)]
    f, h = ft[f"{T}x{H}"]["g2"]["device_fb_ms"], hip[f"{T}x{H}"]["g2"]["device_fb_ms"]
    d, fa = n[("fla", "kernel_fb_ms")], n[("fla_nr", "kernel_fb_ms")]
    v = dict(drift_def=d / o[("fla", "kernel_fb_ms")], drift_fast=fa / o[("fla_nr", "kernel_fb_ms")], df=d / f, ff=fa / f, dh=d / h, fh=fa / h)
    for k, x in v.items(): gm[k].append(x)
    b = lambda x: f"**{x:.2f}**" if x >= 1 else f"{x:.2f}"
    print(f"| {T//1024}K | TP{tp[H]} ({H}) | {d:.2f} | {o[('fla','kernel_fb_ms')]:.2f} | {fa:.2f} | {o[('fla_nr','kernel_fb_ms')]:.2f} | {f:.2f} | {h:.2f} | {b(v['df'])} | {b(v['ff'])} | {b(v['dh'])} | {b(v['fh'])} |")
print("\n### Wall time (ms, events around one fwd+bwd, median of runs); ratios A800 / ours\n")
print("| T | TP (H) | A800 FLA def | A800 FLA fast | flash-train | hip_kda | def / ft | fast / ft |")
print("|---|---|---:|---:|---:|---:|---:|---:|")
for (T, H) in sorted(new):
    n = new[(T, H)]; w = wall.get((T, H))
    if not w: continue
    f, h = st.median(w["ftc"]), st.median(w["g2"])
    d, fa = n[("fla", "fb_ms")], n[("fla_nr", "fb_ms")]
    gm["wdf"].append(d / f); gm["wff"].append(fa / f); gm["wftg2"].append(f / h)
    b = lambda x: f"**{x:.2f}**" if x >= 1 else f"{x:.2f}"
    print(f"| {T//1024}K | TP{tp[H]} ({H}) | {d:.2f} | {fa:.2f} | {f:.2f} | {h:.2f} | {b(d/f)} | {b(fa/f)} |")
g = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))
print("\ngeomean: " + ", ".join(f"{k} {g(v):.3f} [{min(v):.3f}-{max(v):.3f}] (n={len(v)})" for k, v in gm.items()))
