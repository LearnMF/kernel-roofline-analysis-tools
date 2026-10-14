# Extract the executed loop body (one steady-state step, production binary) from SQTT -> pseudo .s for isa_sim
import json, glob, collections, sys
d, disp, nt, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
f = sorted(glob.glob(f"{d}/*thread_trace_{disp}__*.json"))[0]
ev = json.load(open(f))["traceEvents"]
st = collections.defaultdict(list)
for e in ev:
    if e.get("ph") == "X" and "assembly" in e.get("args", {}): st[(e["pid"], e["tid"])].append(e)
s = sorted(max(st.values(), key=len), key=lambda e: (e["ts"], e["args"].get("inst idx", 0)))
asm = [e["args"]["assembly"].strip() for e in s]
cnt = collections.Counter(asm)
# loop marker: the loop-counter increment (s_add_i32 sX, sY, 1) that executes once per step (k * NT times when the
# wave's stream holds k CTAs); fall back to any instruction executed a multiple of NT times
cands = [a for a in asm if a.startswith("s_add_i32") and a.endswith(", 1") and cnt[a] >= nt - 1 and cnt[a] % nt in (0, nt - 1)]
cands = cands or [a for a in asm if cnt[a] >= nt - 1 and cnt[a] % nt in (0, nt - 1)] or [None]
marker = sys.argv[5] if len(sys.argv) > 5 else cands[0]   # optional explicit marker (loop-counter update)
idx = [k for k, a in enumerate(asm) if a == marker]
k0 = idx[len(idx) // 2]
k1 = idx[len(idx) // 2 + 1]
body = asm[k0:k1]
with open(out, "w") as fo:
    fo.write(".LBBX:  ; =>This Inner Loop Header: Depth=1\n")
    for a in body:
        fo.write("\t" + a + "\n")
    fo.write("\ts_cbranch_scc1 .LBBX\n")
print(out, "marker:", marker, "instructions:", len(body))
