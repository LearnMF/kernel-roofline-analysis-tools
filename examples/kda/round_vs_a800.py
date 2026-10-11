"""Per-round A800 comparison table (T = 8K/16K/32K x TP = 1/2/4/8): ours (BW, this round) vs the
A800 FLA 0.5.2 measurement (default and fastest = disable_recompute), ratios "A800 time / ours"
(> 1 = ours faster), wall and device views, plus the change vs a previous round if given.

    python3 round_vs_a800.py --nv nv.jsonl --wall OUT/wall.jsonl --dev OUT/dev.json \
        [--prev-wall prev/wall.jsonl --prev-dev prev/dev.json] [--title "round 8"] [--md out.md]"""
import argparse
import collections
import json
import statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("--nv", required=True)
ap.add_argument("--wall", required=True)
ap.add_argument("--dev", required=True)
ap.add_argument("--prev-wall", default=None)
ap.add_argument("--prev-dev", default=None)
ap.add_argument("--title", default="this round")
ap.add_argument("--md", default=None)
a = ap.parse_args()
SHAPES = [(T, H) for T in (8192, 16384, 32768) for H in (96, 48, 24, 12)]
TP = {96: 1, 48: 2, 24: 4, 12: 8}


def load_wall(p):
    agg = collections.defaultdict(list)
    for line in open(p):
        if line.strip():
            d = json.loads(line)["data"]
            for arm, v in d["res"].items():
                for m, x in v.items():
                    if isinstance(x, (int, float)):
                        agg[(d["T"], d["H"], arm, m)].append(x)
    return {k: st.median(v) for k, v in agg.items()}


def load_dev(p):
    d = json.load(open(p))
    return {(int(k.split("x")[0]), int(k.split("x")[1])): v["g2"]["device_fb_ms"] for k, v in d.items() if "g2" in v}


nv = load_wall(a.nv)
w = load_wall(a.wall)
dv = load_dev(a.dev)
pw = load_wall(a.prev_wall) if a.prev_wall else {}
pd = load_dev(a.prev_dev) if a.prev_dev else {}
L = [f"## {a.title}: ours (BW) vs A800 FLA 0.5.2 -- ratio = A800 time / ours (> 1 = ours faster)", ""]
for view in ("device", "wall"):
    L += [f"### {view} time (ms, one fwd+bwd)", "",
          "| T | TP (heads/card) | A800 FLA default | A800 FLA fastest | ours | A800 default / ours | A800 fastest / ours |"
          + (" ours prev -> now |" if (pd if view == 'device' else pw) else ""),
          "|---|---|---:|---:|---:|---:|---:|" + ("---:|" if (pd if view == 'device' else pw) else "")]
    wins_d = wins_f = 0
    for T, H in SHAPES:
        if view == "device":
            ad, af = nv.get((T, H, "fla", "kernel_fb_ms")), nv.get((T, H, "fla_nr", "kernel_fb_ms"))
            ours, prev = dv.get((T, H)), pd.get((T, H))
        else:
            ad, af = nv.get((T, H, "fla", "fb_ms")), nv.get((T, H, "fla_nr", "fb_ms"))
            ours, prev = w.get((T, H, "g2", "fb_ms")), pw.get((T, H, "g2", "fb_ms"))
        if None in (ad, af, ours):
            continue
        rd, rf = ad / ours, af / ours
        wins_d += rd > 1
        wins_f += rf > 1
        b = lambda r: f"**{r:.2f}**" if r > 1 else f"{r:.2f}"
        row = f"| {T // 1024}K | TP{TP[H]} ({H}) | {ad:.2f} | {af:.2f} | {ours:.2f} | {b(rd)} | {b(rf)} |"
        if prev:
            row += f" {prev:.2f} -> {ours:.2f} ({prev / ours:.3f}x) |"
        L.append(row)
    L += ["", f"ours faster than A800 FLA default on {wins_d}/12 shapes, than the fastest config on {wins_f}/12.", ""]
out = "\n".join(L)
print(out)
if a.md:
    open(a.md, "w").write(out + "\n")
