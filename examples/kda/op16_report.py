"""16-shape operator table vs A800 FLA, in the format of the team's FLA comparison doc.

    python3 op16_report.py --nv nv.jsonl --bw bw/wall.jsonl --bwdev bw/dev_parsed.json

Columns: A800 FLA default / A800 FLA fastest (disable_recompute) / ours (BW G2), and the ratios
"A800 time / ours" (>1 = ours faster).  Two views: wall (CUDA/HIP events around one fwd+bwd, the
protocol of the BW report) and device (kernel sum; A800 CUPTI, BW hipprof)."""
import argparse
import collections
import json
import statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("--nv", required=True)
ap.add_argument("--bw", required=True)
ap.add_argument("--bwdev", default=None)
a = ap.parse_args()


def load(p):
    agg, env = collections.defaultdict(lambda: collections.defaultdict(list)), None
    for line in open(p):
        if line.strip():
            d = json.loads(line)["data"]
            env = d.get("env", env)
            for arm, v in d["res"].items():
                for m, x in v.items():
                    if isinstance(x, (int, float)):
                        agg[(d["T"], d["H"])][(arm, m)].append(x)
    return agg, env


nv, env = load(a.nv)
bw, _ = load(a.bw)
dev = json.load(open(a.bwdev)) if a.bwdev else {}
md = lambda A, k, arm, m: st.median(A[k][(arm, m)]) if A[k][(arm, m)] else float("nan")
shapes = sorted(set(nv) & set(bw), key=lambda k: (k[0], -k[1]))
kt = lambda T: f"{T // 1024}K"
L = [f"A800: {env}", "",
     "## Wall time, single-layer KDA fwd+bwd (ms per card; CUDA/HIP events, median of 7, median of processes)", "",
     "| T | TP (heads/card) | A800 FLA default | A800 FLA fastest | **ours (BW)** | **vs A800 FLA default** | vs A800 FLA fastest | BW FLA default (ref) |",
     "|---|---|---:|---:|---:|---:|---:|---:|"]
wins = 0
for k in shapes:
    T, H = k
    fd, ff, g = md(nv, k, "fla", "fb_ms"), md(nv, k, "fla_nr", "fb_ms"), md(bw, k, "g2", "fb_ms")
    wins += fd / g > 1
    L.append(f"| {kt(T)} | {96 // H} ({H}) | {fd:.2f} | {ff:.2f} | **{g:.2f}** | **{fd / g:.2f}x** | {ff / g:.2f}x | {md(bw, k, 'fla', 'fb_ms'):.2f} |")
L += ["", f"ours faster than A800 FLA default (wall) in {wins}/{len(shapes)} shapes", ""]
if dev:
    L += ["## Device time (kernel sum of one fwd+bwd, ms; A800 CUPTI, BW hipprof)", "",
          "| T | TP (heads/card) | A800 FLA default | A800 FLA fastest | **ours (BW)** | **vs A800 FLA default** | vs A800 FLA fastest |",
          "|---|---|---:|---:|---:|---:|---:|"]
    wd = 0
    for k in shapes:
        T, H = k
        key = f"{T}x{H}"
        if key not in dev or "g2" not in dev[key]:
            continue
        fd, ff, g = md(nv, k, "fla", "kernel_fb_ms"), md(nv, k, "fla_nr", "kernel_fb_ms"), dev[key]["g2"]["device_fb_ms"]
        wd += fd / g > 1
        L.append(f"| {kt(T)} | {96 // H} ({H}) | {fd:.2f} | {ff:.2f} | **{g:.2f}** | **{fd / g:.2f}x** | {ff / g:.2f}x |")
    L += ["", f"ours faster than A800 FLA default (device) in {wd}/{len(shapes)} shapes"]
L += ["", "## Peak memory, single layer fwd+bwd (GiB)", "", "| T | TP | A800 FLA default | A800 FLA fastest | ours (BW) |", "|---|---|---:|---:|---:|"]
for k in shapes:
    T, H = k
    L.append(f"| {kt(T)} | {96 // H} | {md(nv, k, 'fla', 'peak_GiB'):.2f} | {md(nv, k, 'fla_nr', 'peak_GiB'):.2f} | {md(bw, k, 'g2', 'peak_GiB'):.2f} |")
print("\n".join(L))
