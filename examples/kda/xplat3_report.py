"""Three-platform KDA operator table: BW1000 + hip_kda G2 (now, and the session-start production measured
interleaved in the same session) vs FLA 0.5.2 on A800 80GB PCIe and on H20-3e (default and disable_recompute
'fastest').  Device time = kernel sum per fwd+bwd (BW hipprof; NV CUPTI); wall = events around one fwd+bwd.
Ratios are 'other / ours' (> 1 = ours faster).  Bandwidth-normalized ratio = ratio x (measured HBM read BW of the
other GPU / BW1000's): > 1 = ours does more per byte of bandwidth.
    python3 xplat3_report.py --a800 nv.jsonl --h20 h20.jsonl --dev dev.json --dev-prev dev_prev.json --wall wall.jsonl"""
import argparse
import collections
import json
import math
import statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("--a800", required=True)
ap.add_argument("--h20", required=True)
ap.add_argument("--dev", required=True)
ap.add_argument("--dev-prev", default=None)
ap.add_argument("--wall", required=True)
a = ap.parse_args()
BW_READ = {"bw": 1339.0, "a800": 1771.0, "h20": 3821.0}   # measured HBM read GB/s (nv_peaks.py / bw microbench)
TP = {12: "TP8", 24: "TP4", 48: "TP2", 96: "TP1"}


def load_nv(p):
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for line in open(p):
        if line.strip():
            d = json.loads(line)["data"]
            for arm, v in d["res"].items():
                for m, x in v.items():
                    if isinstance(x, (int, float)):
                        agg[(d["T"], d["H"])][(arm, m)].append(x)
    return agg


a8, h2 = load_nv(a.a800), load_nv(a.h20)
wall = load_nv(a.wall)
dev = json.load(open(a.dev))
devp = json.load(open(a.dev_prev)) if a.dev_prev else {}
shapes = sorted(a8, key=lambda s: (s[0], s[1]))


def med(agg, s, arm, m):
    v = agg.get(s, {}).get((arm, m))
    return st.median(v) if v else float("nan")


def ratio(x, y):
    return x / y if y and not math.isnan(x) and not math.isnan(y) else float("nan")


def f(x, d=2):
    return "--" if math.isnan(x) else f"{x:.{d}f}"


def b(x):
    return "--" if math.isnan(x) else (f"**{x:.2f}**" if x >= 1.0 else f"{x:.2f}")


print("### Device time (ms, one fwd+bwd) and ratios other/ours (> 1 = ours faster)\n")
print("| T | TP (H/card) | ours now | ours session start | gain | A800 FLA def | A800 FLA fast | H20 FLA def | H20 FLA fast "
      "| A800 def / ours | A800 fast / ours | H20 def / ours | H20 fast / ours |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
geo = collections.defaultdict(list)
for s in shapes:
    T, H = s
    o = dev.get(f"{T}x{H}", {}).get("g2", {}).get("device_fb_ms", float("nan"))
    op = devp.get(f"{T}x{H}", {}).get("g2", {}).get("device_fb_ms", float("nan"))
    ad, af = med(a8, s, "fla", "kernel_fb_ms"), med(a8, s, "fla_nr", "kernel_fb_ms")
    hd, hf = med(h2, s, "fla", "kernel_fb_ms"), med(h2, s, "fla_nr", "kernel_fb_ms")
    r = [ratio(ad, o), ratio(af, o), ratio(hd, o), ratio(hf, o)]
    for k, x in zip(("a8d", "a8f", "h2d", "h2f"), r):
        if not math.isnan(x):
            geo[k].append(x)
            if T >= 16384 and H <= 48:
                geo[k + "_prio"].append(x)
    print(f"| {T // 1024}K | {TP[H]} ({H}) | {f(o)} | {f(op)} | {f(ratio(op, o), 3)} | {f(ad)} | {f(af)} | {f(hd)} | {f(hf)} | "
          + " | ".join(b(x) for x in r) + " |")


def gm(v):
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


print(f"\ngeomean over 16 shapes: A800 def/ours {gm(geo['a8d']):.3f}, A800 fast/ours {gm(geo['a8f']):.3f}, "
      f"H20 def/ours {gm(geo['h2d']):.3f}, H20 fast/ours {gm(geo['h2f']):.3f}")
print(f"geomean over the 9 priority shapes (16K-64K, TP>1): A800 def/ours {gm(geo['a8d_prio']):.3f}, "
      f"H20 def/ours {gm(geo['h2d_prio']):.3f}, H20 fast/ours {gm(geo['h2f_prio']):.3f}")

print("\n### Wall time (ms, events around one fwd+bwd) and ratios other/ours\n")
print("| T | TP | ours | A800 FLA def | A800 FLA fast | H20 FLA def | H20 FLA fast | A800 def / ours | H20 def / ours |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
for s in shapes:
    T, H = s
    o = med(wall, s, "g2", "fb_ms")
    ad, af = med(a8, s, "fla", "fb_ms"), med(a8, s, "fla_nr", "fb_ms")
    hd, hf = med(h2, s, "fla", "fb_ms"), med(h2, s, "fla_nr", "fb_ms")
    print(f"| {T // 1024}K | {TP[H]} | {f(o)} | {f(ad)} | {f(af)} | {f(hd)} | {f(hf)} | {b(ratio(ad, o))} | {b(ratio(hd, o))} |")

print("\n### Bandwidth-normalized device-time ratio (other/ours x other_BW/BW1000_BW; > 1 = more work per byte/s)\n")
print(f"measured HBM read: BW1000 {BW_READ['bw']:.0f}, A800 {BW_READ['a800']:.0f}, H20 {BW_READ['h20']:.0f} GB/s\n")
print("| T | TP | vs A800 FLA def | vs H20 FLA def | vs H20 FLA fast |")
print("|---|---|---:|---:|---:|")
for s in shapes:
    T, H = s
    o = dev.get(f"{T}x{H}", {}).get("g2", {}).get("device_fb_ms", float("nan"))
    ad, hd, hf = med(a8, s, "fla", "kernel_fb_ms"), med(h2, s, "fla", "kernel_fb_ms"), med(h2, s, "fla_nr", "kernel_fb_ms")
    print(f"| {T // 1024}K | {TP[H]} | {f(ratio(ad, o) * BW_READ['a800'] / BW_READ['bw'])} | "
          f"{f(ratio(hd, o) * BW_READ['h20'] / BW_READ['bw'])} | {f(ratio(hf, o) * BW_READ['h20'] / BW_READ['bw'])} |")
