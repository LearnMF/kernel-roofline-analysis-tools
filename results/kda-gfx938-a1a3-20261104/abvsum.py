# abvsum.py REFTAG KEYS FILES... : median per-kernel time ratio ref / variant per shape
import json, sys, collections, statistics as st
ref, keys = sys.argv[1], sys.argv[2].split(",")
rows = [json.loads(l) for f in sys.argv[3:] for l in open(f) if l.startswith("{")]
by = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows: by[(r["T"], r["H"])][r["tag"]].append(r)
def kt(r, k): return sum(v for n, v in r["per"].items() if k in n)
for s, d in sorted(by.items()):
    line = f"{s[0]}/{s[1]}:"
    R = d[ref]
    for tag in [t for t in d if t != ref]:
        X = d[tag]; parts = []
        for k in keys:
            a, b = st.median(kt(r, k) for r in R), st.median(kt(r, k) for r in X)
            parts.append(f"{k} {a:.0f}->{b:.0f} ({a / b:.3f}x)" if b else f"{k} {a:.0f}->0")
        ta, tb = st.median(r["tot_avg"] for r in R), st.median(r["tot_avg"] for r in X)
        parts.append(f"tot {ta / 1000:.3f}->{tb / 1000:.3f} ({ta / tb:.3f}x)")
        line += f"\n   {tag}: " + " | ".join(parts)
    print(line)
