"""Combine guarded 24-shape passes (fb_guard.sh rows: a16 = 2631309, bc24 = c1fc134) on BW1300 (p8c nodes).
A (pass, shape) block of 8 rows is accepted only if: 8 rows present, no row has post_busy, no pre_wait_s > 30 (a foreign
job held the GPU just before), and the unchanged forward reads within 1.5% (fwd ratio is the noise check).
python3 t24_combine.py pass1 pass2 [pass3 ...] -> per-shape table from the accepted blocks + list of shapes to rerun."""
import collections, glob, json, math, statistics as st, sys

SHAPES = [(t, h) for t in (1024, 2048, 4096, 8192, 16384, 32768) for h in (64, 32, 16, 8)]
blocks = collections.defaultdict(list)          # (T, H) -> [(pass, rows)]
for p in sys.argv[1:]:
    by = collections.defaultdict(list)          # one block = one shape on one GPU of one host
    for f in glob.glob(f"{p}/t24_*.jsonl"):
        for l in open(f):
            l = l.strip()
            if l.startswith("{") and '"T"' in l:
                r = json.loads(l); by[(r["T"], r["H"], r["host"], r["gpu"])].append(r)
    for (T, H, host, gpu), rows in by.items(): blocks[(T, H)].append((f"{p}/{host}.{gpu}", rows))

med = lambda rows, tag, key: st.median(key(r) for r in rows if r["tag"] == tag)
fwd = lambda r: r["fwd_ms"]; bwd = lambda r: r["bwd_ms"]; tot = lambda r: r["fwd_ms"] + r["bwd_ms"]
out, rerun, gm = [], [], collections.defaultdict(list)
for T, H in SHAPES:
    acc, why = [], []
    for p, rows in blocks.get((T, H), []):
        bad = []
        if len(rows) != 8 or {r["tag"] for r in rows} != {"a16", "bc24"}: bad.append("rows")
        else:
            if any(r["post_busy"] for r in rows): bad.append("post_busy")
            if any(r["pre_wait_s"] > 30 for r in rows): bad.append("waited")
            rf = med(rows, "a16", fwd) / med(rows, "bc24", fwd)
            if abs(rf - 1) > 0.015: bad.append(f"fwd {rf:.3f}")
        (why if bad else acc).append(f"{p}: {','.join(bad)}" if bad else rows)
    if not acc:
        rerun.append((T, H)); out.append(f"| {T}/{H} | -- | -- | -- | -- | rerun ({'; '.join(why)}) |"); continue
    rows = [r for b in acc for r in b]
    ra = {k: med(rows, "a16", f) for k, f in (("f", fwd), ("b", bwd), ("t", tot))}
    rb = {k: med(rows, "bc24", f) for k, f in (("f", fwd), ("b", bwd), ("t", tot))}
    for k in "fbt": gm[k].append(ra[k] / rb[k])
    per_pass = " / ".join(f"{med(b, 'a16', tot) / med(b, 'bc24', tot):.3f}" for b in acc)
    out.append(f"| {T}/{H} | {ra['t']:.3f} -> {rb['t']:.3f} | {ra['f'] / rb['f']:.3f} | {ra['b'] / rb['b']:.3f} | "
               f"**{ra['t'] / rb['t']:.3f}** | {len(acc)} block(s): {per_pass}" + (f"; rejected {'; '.join(why)}" if why else "") + " |")
print("| T/H | total ms a16 -> bc24 | fwd | bwd | total | accepted blocks (per-block total ratio) |")
print("|---|---|---|---|---|---|")
print("\n".join(out))
g = lambda xs: math.prod(xs) ** (1 / len(xs)) if xs else float("nan")
print(f"\ngeomean over {len(gm['t'])} shapes: fwd {g(gm['f']):.4f}, bwd {g(gm['b']):.4f}, total {g(gm['t']):.4f}")
print("rerun:", " ".join(f"{t}:{h}" for t, h in rerun) or "none")
