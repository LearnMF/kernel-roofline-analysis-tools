"""Operator-level dataflow graph from a kra optrace (launcher interface capture).

Each launcher output at a pointer creates a new tensor version; later inputs at that pointer read
the latest version.  Tensors never produced inside the trace are external (model inputs, or state
saved by the forward).  Reports, per intermediate tensor: producer, consumers, bytes written and
bytes read, and the totals that separate the irreducible external traffic (read each model input
once, write each model output once) from intermediate round trips.  This is the input of the
fusion cost model (rule R29: whole-operator intensity vs machine balance).

    python3 dataflow_graph.py optrace.json [--flops-gf F] [--json out.json]"""
import argparse
import collections
import json

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--flops-gf", type=float, default=None, help="model FLOPs of fwd+bwd (GFLOP) for intensity")
ap.add_argument("--json", default=None)
a = ap.parse_args()
j = json.load(open(a.trace))
calls = j["calls"]
ver = {}                       # ptr -> current tensor id
tensors = {}                   # id -> dict
reads = collections.defaultdict(list)
ext_n = 0


def tid_of(ptr, rec, launcher, seq, is_out, k):
    global ext_n
    if is_out:
        t = f"{launcher}.o{k}"
        tensors[t] = dict(id=t, producer=launcher, seq=seq, bytes=rec["bytes"], shape=rec["shape"],
                          dtype=rec["dtype"], external=False)
        ver[ptr] = t
        return t
    if ptr not in ver:
        t = f"ext{ext_n}"
        ext_n += 1
        tensors[t] = dict(id=t, producer=None, seq=-1, bytes=rec["bytes"], shape=rec["shape"],
                          dtype=rec["dtype"], external=True)
        ver[ptr] = t
    return ver[ptr]


for c in calls:
    for k, r in enumerate(c["inputs"]):
        if r.get("dup"):
            continue
        t = tid_of(r["ptr"], r, c["launcher"], c["seq"], False, k)
        reads[t].append((c["launcher"], r["bytes"]))
    for k, r in enumerate(c["outputs"]):
        if r.get("dup"):
            continue
        tid_of(r["ptr"], r, c["launcher"], c["seq"], True, k)

ext_in = sum(t["bytes"] for t in tensors.values() if t["external"])
inter = [t for t in tensors.values() if not t["external"]]
consumed = {t["id"] for t in inter if reads.get(t["id"])}
final_out = [t for t in inter if t["id"] not in consumed]       # never read again: operator outputs
inter_rt = [t for t in inter if t["id"] in consumed]
w_inter = sum(t["bytes"] for t in inter_rt)
r_inter = sum(b for t in inter_rt for _, b in reads[t["id"]])
r_ext = sum(b for t in tensors.values() if t["external"] for _, b in reads[t["id"]])
w_final = sum(t["bytes"] for t in final_out)
tot = sum(c["in_bytes"] + c["out_bytes"] for c in calls)
MB = 1e6
print(f"trace {a.trace}  meta {j['meta']}")
print(f"launcher interface traffic total      {tot / MB:10.1f} MB")
print(f"  external inputs, first read (min)    {ext_in / MB:10.1f} MB   (all reads of them: {r_ext / MB:.1f} MB)")
print(f"  operator outputs written (min)       {w_final / MB:10.1f} MB")
print(f"  intermediates written                {w_inter / MB:10.1f} MB   ({len(inter_rt)} tensors)")
print(f"  intermediates read                   {r_inter / MB:10.1f} MB")
mn = ext_in + w_final
print(f"  => irreducible (fully fused) traffic {mn / MB:10.1f} MB ; current / irreducible = {tot / mn:.2f}x")
if a.flops_gf:
    print(f"  intensity: current {a.flops_gf * 1e9 / tot:6.1f} FLOP/B ; fully fused {a.flops_gf * 1e9 / mn:6.1f} FLOP/B")
print("\nintermediate tensors by round-trip bytes (write + all reads):")
rows = sorted(inter_rt, key=lambda t: -(t["bytes"] + sum(b for _, b in reads[t["id"]])))
for t in rows:
    rs = reads[t["id"]]
    rt = t["bytes"] + sum(b for _, b in rs)
    cons = ",".join(f"{l}" for l, _ in rs)
    print(f"  {rt / MB:8.1f} MB  w {t['bytes'] / MB:6.1f}  {t['producer']:>12s} -> {cons:40s} {t['dtype']:9s} {t['shape']}")
print("\nexternal inputs (first read):")
for t in tensors.values():
    if t["external"]:
        cons = ",".join(sorted({l for l, _ in reads[t["id"]]}))
        print(f"  {t['bytes'] / MB:8.1f} MB  read by {len(reads[t['id']])}x [{cons}]  {t['dtype']} {t['shape']}")
print("\noperator outputs (never re-read):")
for t in final_out:
    print(f"  {t['bytes'] / MB:8.1f} MB  {t['producer']:>12s}  {t['dtype']} {t['shape']}")
if a.json:
    json.dump({"total": tot, "ext_in": ext_in, "final_out": w_final, "inter_w": w_inter, "inter_r": r_inter,
               "tensors": [dict(t, reads=reads[t["id"]]) for t in tensors.values()]}, open(a.json, "w"), indent=1)
