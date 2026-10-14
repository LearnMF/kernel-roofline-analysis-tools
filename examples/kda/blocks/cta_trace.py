# Extract one CTA's per-wave dynamic instruction sequences of a chunk-parallel kernel from SQTT -> JSON
# (waves of one CTA = wave segments in different streams starting within 64 cycles of each other)
import json, glob, collections, sys
d, disp, W, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
segs = []
for f in sorted(glob.glob(f"{d}/*thread_trace_{disp}__*.json")):
    ev = json.load(open(f))["traceEvents"]
    st = collections.defaultdict(list)
    for e in ev:
        if e.get("ph") == "X" and "assembly" in e.get("args", {}): st[(f, e["pid"], e["tid"])].append(e)
    for key, s in st.items():
        s.sort(key=lambda e: (e["ts"], e["args"].get("inst idx", 0)))
        cur = []
        for e in s:
            cur.append(e)
            if e["name"] == "s_endpgm":
                segs.append((key, cur[0]["ts"], cur[-1]["ts"], [x["args"]["assembly"].strip() for x in cur]))
                cur = []
segs.sort(key=lambda x: x[1])
best = None
for i in range(len(segs) // 3, len(segs)):
    grp = [segs[i]] + [s for s in segs if s is not segs[i] and abs(s[1] - segs[i][1]) <= 64 and s[0][:2] == segs[i][0][:2]]
    keys = {g[0] for g in grp}
    if len(keys) >= W:
        best = sorted(grp, key=lambda g: g[0])[:W]
        break
if best is None:   # fall back: W segments with the most common length
    best = segs[len(segs) // 2: len(segs) // 2 + W]
json.dump({"waves": [g[3] for g in best], "durations": [g[2] - g[1] for g in best]}, open(out, "w"))
print(out, "waves", len(best), "instructions", [len(g[3]) for g in best], "durations", [g[2] - g[1] for g in best])
