"""Stall attribution by instruction text (+ the 2 preceding instructions for waits / barriers): issue + stall cycles.
    python3 stall_ctx.py DIR [MAXF] [TOP]"""
import collections, glob, json, re, sys
d = sys.argv[1]; maxf = int(sys.argv[2]) if len(sys.argv) > 2 else 4; top = int(sys.argv[3]) if len(sys.argv) > 3 else 40
files = sorted(glob.glob(f"{d}/*thread_trace_*__*.json"))
disp = sorted({re.search(r"thread_trace_(\d+)__", f).group(1) for f in files}, key=int)[-1]
cnt, stall, iss = collections.Counter(), collections.Counter(), collections.Counter()
cls = collections.Counter(); nw = 0
def kind(s):
    op = s.split()[0]
    if op == "s_waitcnt": return "wait_" + ("vm" if "vmcnt" in s else "lgkm")
    if op == "s_barrier": return "barrier"
    for p, n in (("global_load", "vmem_ld"), ("global_store", "vmem_st"), ("buffer_", "vmem"), ("matrix_load", "mls"), ("ds_", "lds"),
                 ("v_mmac", "mmac"), ("s_", "salu"), ("v_", "valu")):
        if op.startswith(p): return n
    return "other"
for f in [f for f in files if f"thread_trace_{disp}__" in f][:maxf]:
    waves = collections.defaultdict(list)
    for e in json.load(open(f))["traceEvents"]:
        a = e.get("args", {})
        if e.get("ph") == "X" and "assembly" in a: waves[(e["pid"], e["tid"])].append((a["time"], a.get("clks", 4), a["assembly"].strip()))
    for xs in waves.values():
        xs.sort(); nw += 1; prev = xs[0][0]; h = ["", ""]
        for t, c, s in xs:
            k = kind(s)
            key = (h[0], h[1], s) if k in ("wait_vm", "wait_lgkm", "barrier") else ("", "", s)
            st = max(0, t - prev)
            cnt[key] += 1; iss[key] += c; stall[key] += st; cls[k] += c + st
            prev = t + c; h = [h[1], s]
tot = sum(cls.values())
print(f"{d}: {nw} waves, {tot / nw:.0f} cycles / wave")
print("by class: " + "  ".join(f"{k} {100 * v / tot:.1f}%" for k, v in cls.most_common()))
print(f"{'execs/w':>8s} {'stall%':>7s} {'issue%':>7s}  instruction   <- preceded by")
for key, s in sorted(stall.items(), key=lambda kv: -kv[1])[:top]:
    ctx = f"   <- {key[0][:38]} | {key[1][:38]}" if key[0] or key[1] else ""
    print(f"{cnt[key] / nw:8.1f} {100 * s / tot:6.2f}% {100 * iss[key] / tot:6.2f}%  {key[2][:52]}{ctx}")
