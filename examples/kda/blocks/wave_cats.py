# per-wave issue-time accounting of one CTA trace from SQTT (same categories as sqtt_step.py), averaged over waves
import json, glob, collections, sys
d = sys.argv[1]
disp = sorted({f.split("thread_trace_")[1].split("__")[0] for f in glob.glob(f"{d}/*thread_trace_*__*.json")})[-1]
def cls(a):
    op = a.split()[0]
    if op.startswith("v_mmac"): return "mmac"
    if op.startswith(("v_exp","v_rcp","v_log","v_sqrt","v_rsq")): return "trans"
    if op.startswith("v_"): return "valu"
    if op.startswith("ds_"): return "lds"
    if op.startswith(("global_","buffer_")): return "vmem"
    return "salu"
tot = collections.Counter(); n = 0; dur = []
for f in sorted(glob.glob(f"{d}/*thread_trace_{disp}__*.json"))[:4]:
    ev = json.load(open(f))["traceEvents"]
    st = collections.defaultdict(list)
    for e in ev:
        if e.get("ph") == "X" and "assembly" in e.get("args", {}): st[(e["pid"], e["tid"])].append(e)
    for s in st.values():
        s.sort(key=lambda e: (e["ts"], e["args"].get("inst idx", 0)))
        cur = []
        for e in s:
            cur.append(e)
            if e["name"] != "s_endpgm": continue
            w, cur = cur, []
            if len(w) < 50: continue
            n += 1; dur.append(w[-1]["ts"] - w[0]["ts"])
            for k in range(1, len(w)):
                x, p = w[k], w[k-1]
                gap = max(0, x["ts"] - p["ts"] - p["args"].get("clks", 4))
                a = x["args"]["assembly"]
                key = "wait" if x["name"] == "s_waitcnt" else "barrier" if x["name"] == "s_barrier" else "issue:" + cls(a)
                tot[key] += gap; tot["busy:" + cls(p["args"]["assembly"])] += p["args"].get("clks", 4)
dur.sort()
print(f"{d}: {n} waves, median wave duration {dur[len(dur)//2]} cycles")
print({k: round(v / n) for k, v in sorted(tot.items(), key=lambda kv: -kv[1])})
