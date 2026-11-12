import json, sys, glob, collections, statistics as st
rows = [json.loads(l) for f in sys.argv[2:] for l in open(f) if l.startswith("{")]
R = sys.argv[1]
by = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows: by[(r["T"], r["H"])][r["tag"]].append(r)
gm = collections.defaultdict(list)
for s, d in sorted(by.items()):
    V = [t for t in d if t != R][0]
    a, b = d[R], d[V]
    f = lambda xs, k: st.median(x[k] for x in xs)
    tot = lambda xs: st.median(x["fwd_ms"] + x["bwd_ms"] for x in xs)
    rf, rb, rt = f(a, "fwd_ms") / f(b, "fwd_ms"), f(a, "bwd_ms") / f(b, "bwd_ms"), tot(a) / tot(b)
    gm["f"].append(rf); gm["b"].append(rb); gm["t"].append(rt)
    print(f"{s[0]}/{s[1]}: n {len(a)}/{len(b)} fwd {f(a,'fwd_ms'):.3f}->{f(b,'fwd_ms'):.3f} ({rf:.3f}x) bwd {f(a,'bwd_ms'):.3f}->{f(b,'bwd_ms'):.3f} ({rb:.3f}x) total {tot(a):.3f}->{tot(b):.3f} ({rt:.3f}x)  bwd peak MB {a[0]['bwd_peak_mb']} -> {b[0]['bwd_peak_mb']}")
g = lambda xs: __import__("math").prod(xs) ** (1 / len(xs))
print(f"geomean fwd {g(gm['f']):.4f} bwd {g(gm['b']):.4f} total {g(gm['t']):.4f}")
