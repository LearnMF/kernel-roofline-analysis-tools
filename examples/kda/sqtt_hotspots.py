"""Instruction-level wait attribution from hipprof SQTT (--sqtt --sqtt-type 1, --output-type 0).

Each wave's instruction records (ts = the cycle the instruction ISSUED, clks, assembly) are in program
order, and an s_waitcnt / s_barrier is recorded when it is released -- so the gap BEFORE a record is
the time the wave could not issue it.  At an s_waitcnt the wave waited for vector memory (vmcnt:
FLAT/BUFFER loads+stores, retired in issue order) and/or LDS/SMEM (lgkmcnt); replaying the wave's
outstanding queues tells which memory instructions had to retire, and the wait is charged to them.
A static waitcnt SITE is keyed by its assembly plus the two instructions before it.

    python3 sqtt_hotspots.py <dir-with-thread_trace_*.json> [--dispatch 0] [--top 25] [--json out]"""
import argparse
import collections
import glob
import json
import re

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--dispatch", default="0")
ap.add_argument("--top", type=int, default=25)
ap.add_argument("--json", default=None)
a = ap.parse_args()

files = sorted(glob.glob(f"{a.dir}/*thread_trace_{a.dispatch}__*.json"))
assert files, "no trace files"
VM = ("FLAT_RD", "FLAT_WR", "BUF_RD", "BUF_WR", "VMEM_RD", "VMEM_WR", "BUFFER_RD", "BUFFER_WR")
LGKM = ("LDS", "SMEM_RD", "SMEM_WR", "GDS")
wcnt = re.compile(r"(vmcnt|lgkmcnt)\((\d+)\)")

tot = collections.Counter()
by_type = collections.Counter()
pre = collections.Counter()
site = collections.Counter()
site_n = collections.Counter()
drained = collections.Counter()
drained_n = collections.Counter()
waves = 0
for f in files:
    ev = json.load(open(f))["traceEvents"]
    streams = collections.defaultdict(list)
    for e in ev:
        if e.get("ph") == "X" and "assembly" in e.get("args", {}):
            streams[(e["pid"], e["tid"])].append(e)
    for s in streams.values():
        s.sort(key=lambda e: (e["ts"], e["args"].get("inst idx", 0)))
        cur = []
        for e in s:
            cur.append(e)
            if e["name"] != "s_endpgm":
                continue
            w, cur = cur, []
            waves += 1
            vmq, lgq = [], []
            for k, x in enumerate(w):
                t = x["args"].get("inst type", "")
                asm = x["args"]["assembly"]
                clk = x["args"].get("clks", 4)
                by_type[t] += 1
                tot["issue"] += clk
                gap = 0 if k == 0 else max(0, x["ts"] - w[k - 1]["ts"] - w[k - 1]["args"].get("clks", 4))
                key = (asm, w[k - 1]["args"]["assembly"] if k else "", w[k - 2]["args"]["assembly"] if k > 1 else "")
                if x["name"] == "s_waitcnt":
                    tot["waitcnt"] += gap
                    site[key] += gap
                    site_n[key] += 1
                    req = dict((m.group(1), int(m.group(2))) for m in wcnt.finditer(asm))
                    ret = []
                    if "vmcnt" in req and len(vmq) > req["vmcnt"]:
                        n = len(vmq) - req["vmcnt"]
                        ret += [("vm", y) for y in vmq[:n]]
                        vmq = vmq[n:]
                    if "lgkmcnt" in req and len(lgq) > req["lgkmcnt"]:
                        n = len(lgq) - req["lgkmcnt"]
                        ret += [("lgkm", y) for y in lgq[:n]]
                        lgq = lgq[n:]
                    for r_ in ret:
                        drained[r_] += gap / len(ret)
                        drained_n[r_] += 1
                elif x["name"] == "s_barrier":
                    tot["barrier"] += gap
                    site[key] += gap
                    site_n[key] += 1
                else:
                    tot["issue_delay"] += gap
                    pre[t] += gap
                if t in VM:
                    vmq.append(asm)
                elif t in LGKM:
                    lgq.append(asm)
allc = sum(tot.values())
print(f"{len(files)} SE trace file(s), {waves} complete waves; wave-cycles {allc}")
for k in ("issue", "waitcnt", "barrier", "issue_delay"):
    print(f"  {k:12s} {tot[k]:12d}  {100 * tot[k] / allc:5.1f}%")
print(f"\nwait sites (top {a.top}): % of wave-cycles, cycles, count, site = wait <- prev <- prev2")
for key, c in site.most_common(a.top):
    print(f"  {100 * c / allc:5.1f}%  {c:10d}  x{site_n[key]:5d}  {key[0]}   <- {key[1]}   <- {key[2]}")
print(f"\nwait charged to the drained memory instructions (top {a.top}):")
for (cls, asm), c in drained.most_common(a.top):
    print(f"  {100 * c / allc:5.1f}%  {c:12.0f}  x{drained_n[(cls, asm)]:6d}  [{cls}] {asm}")
print("\nissue delay outside waits, by type:", {t: f"{100 * c / allc:.1f}%" for t, c in pre.most_common(8)})
print("issued instructions by type:", dict(by_type.most_common()))
if a.json:
    json.dump({"waves": waves, "cycles": dict(tot),
               "sites": [{"wait": k[0], "prev": k[1], "prev2": k[2], "cycles": v, "n": site_n[k]} for k, v in site.most_common()],
               "drained": [{"cls": c, "asm": s_, "cycles": v} for (c, s_), v in drained.most_common()],
               "by_type": dict(by_type)}, open(a.json, "w"), indent=1)
