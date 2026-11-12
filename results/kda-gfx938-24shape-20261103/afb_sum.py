import json, sys, collections, statistics as st, math
# afb_sum.py AFB_ALL.jsonl H20_FLA.jsonl : per-shape median fwd / bwd / total of REF (spa = 5ebce6f) and VAR (va2 = 69de6cf)
d = collections.defaultdict(list)
for l in open(sys.argv[1]):
    r = json.loads(l); d[(r["T"], r["H"], r["tag"])].append(r)
h = collections.defaultdict(list)
for l in open(sys.argv[2]):
    r = json.loads(l); h[(r["T"], r["H"])].append(r)
med = lambda rs, k: st.median(x[k] for x in rs)
shapes = sorted({(T, H) for T, H, _ in d}, key=lambda x: (x[0], -x[1]))
print("| T/H | 5ebce6f fwd / bwd / total | 69de6cf fwd / bwd / total | 69de6cf vs 5ebce6f fwd / bwd / total | H20 FLA fwd / bwd / total | 69de6cf vs H20 fwd / bwd / total | gap to H20 (ms) fwd / bwd |")
print("|---|---|---|---|---|---|---|")
g = collections.defaultdict(list); ge = 0; n = 0; fgap = bgap = 0.0
for T, H in shapes:
    a, b = d[(T, H, "spa")], d[(T, H, "va2")]
    af, ab = med(a, "fwd_ms"), med(a, "bwd_ms"); bf, bb = med(b, "fwd_ms"), med(b, "bwd_ms")
    hf, hb = med(h[(T, H)], "fwd_ms"), med(h[(T, H)], "bwd_ms")
    s = (af / bf, ab / bb, (af + ab) / (bf + bb)); x = (hf / bf, hb / bb, (hf + hb) / (bf + bb))
    for k, v in zip(("sf", "sb", "st", "xf", "xb", "xt"), s + x): g[k].append(math.log(v))
    ge += x[2] >= 1; n += 1
    fgap += max(bf - hf, 0); bgap += max(bb - hb, 0)
    print(f"| {T}/{H} | {af:.3f} / {ab:.3f} / {af+ab:.3f} | {bf:.3f} / {bb:.3f} / {bf+bb:.3f} | {s[0]:.3f} / {s[1]:.3f} / **{s[2]:.3f}** "
          f"| {hf:.3f} / {hb:.3f} / {hf+hb:.3f} | {x[0]:.2f} / {x[1]:.2f} / **{x[2]:.3f}** | {bf-hf:+.3f} / {bb-hb:+.3f} |")
gm = {k: math.exp(sum(v) / len(v)) for k, v in g.items()}
print(f"\n69de6cf vs 5ebce6f geomean: fwd {gm['sf']:.4f}, bwd {gm['sb']:.4f}, total {gm['st']:.4f}; "
      f"range total {min(math.exp(v) for v in g['st']):.3f}-{max(math.exp(v) for v in g['st']):.3f}")
print(f"69de6cf vs H20 FLA default geomean: fwd {gm['xf']:.3f}, bwd {gm['xb']:.3f}, total {gm['xt']:.3f}; total >= 1 on {ge}/{n}")
print(f"sum over shapes where behind H20: fwd deficit {fgap:.2f} ms, bwd deficit {bgap:.2f} ms")
