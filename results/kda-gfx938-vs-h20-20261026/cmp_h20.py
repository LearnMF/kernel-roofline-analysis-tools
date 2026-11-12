import json, statistics as st, collections, sys
h = collections.defaultdict(list); g = collections.defaultdict(list)
for l in open("/tmp/h20_fb.jsonl"):
    r = json.loads(l)
    if "err" not in r: h[(r["T"], r["H"])].append(r)
for l in open("/tmp/g938_fb.jsonl"):
    r = json.loads(l); g[(r["T"], r["H"])].append(r)
print(f"{'T/H':>9s} | H20 FLA fwd   bwd   tot | gfx938 FT fwd  bwd   tot | gfx938/H20 speedup fwd bwd tot | target tot (H20/1.5) | need")
import math; lg = []
for T in (1024, 2048, 4096, 8192, 16384, 32768):
    for H in (64, 32, 16, 8):
        if (T, H) not in g: continue
        gf, gb = st.median(r["fwd_ms"] for r in g[(T, H)]), st.median(r["bwd_ms"] for r in g[(T, H)])
        if (T, H) in h:
            hf, hb = st.median(r["fwd_ms"] for r in h[(T, H)]), st.median(r["bwd_ms"] for r in h[(T, H)])
            ht = hf + hb; gt = gf + gb; lg.append(math.log(ht / gt))
            print(f"{T:6d}/{H:2d} | {hf:7.3f} {hb:7.3f} {ht:7.3f} | {gf:7.3f} {gb:7.3f} {gt:7.3f} | {hf/gf:5.2f}x {hb/gb:5.2f}x {ht/gt:5.2f}x | {ht/1.5:7.3f} | {gt/(ht/1.5):4.2f}x")
        else:
            print(f"{T:6d}/{H:2d} | H20 OOM (other users hold 132 GB) | {gf:7.3f} {gb:7.3f} {gf+gb:7.3f}")
if lg: print(f"geomean gfx938 vs H20 (total): {math.exp(sum(lg)/len(lg)):.3f}x over {len(lg)} shapes")
