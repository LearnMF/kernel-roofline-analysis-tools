"""Critical-path attribution of a barrier-synchronised recurrence step from hipprof SQTT (--sqtt-type 1).

One trace file = one CU = one CTA (TP8 regime: one CTA per CU).  The step time of the CTA is the chain of events
between two releases of the same barrier.  This walks that chain BACKWARD from the release of barrier b1 to the
release of barrier b0 (steady window, 15%..85% of the steps), choosing at every instruction x of the current wave:

  * x = s_barrier with a stall   -> the release waited for the LAST-ARRIVING wave: jump to that wave's arrival
                                     (release - arrival is charged to "barrier:sync").
  * x = s_waitcnt with a stall   -> the wait ended when the youngest REQUIRED memory op returned (in-order vmcnt /
                                     lgkmcnt queues replayed from the issue stream): jump to that op's issue; the
                                     interval is charged to "lat:<op>" (memory latency on the chain).
  * x reads an MMAC result and stalled -> jump to that MMAC; charged to "dep:mmac-><class of x>".
  * any other stall              -> the wave itself could not issue x (pipe / queue busy): "stall:<class of x>".
  * no stall                     -> the previous instruction's issue cycles are on the chain: "issue:<class>".

Everything issued in the shadow of a jump is off the critical path.  Output: per-step cycles by category and the
top critical instructions (by PC) with their share.
    python3 cp_step.py <trace_dir> --dispatch 2 [--thresh 8] [--json out.json]"""
import argparse
import bisect
import collections
import glob
import json
import re

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--dispatch", default="2")
ap.add_argument("--thresh", type=int, default=8, help="extra cycles before an instruction that count as a stall")
ap.add_argument("--mmac-window", type=int, default=160, help="max cycles from an MMAC issue to a dependent read")
ap.add_argument("--json", default=None)
ap.add_argument("--top", type=int, default=30)
a = ap.parse_args()

VREG = re.compile(r"\bv\[(\d+):(\d+)\]|\bv(\d+)\b")


def vregs(ops):
    out = set()
    for m in VREG.finditer(ops):
        if m.group(3) is not None:
            out.add(int(m.group(3)))
        else:
            out.update(range(int(m.group(1)), int(m.group(2)) + 1))
    return out


def cls(op):
    if op.startswith("v_mmac"):
        return "mmac"
    if op.startswith(("v_exp", "v_rcp", "v_log", "v_sqrt", "v_rsq")):
        return "trans"
    if op.startswith("v_readfirstlane") or op.startswith("v_readlane") or op.startswith("v_writelane"):
        return "valu-xlane"
    if op.startswith("v_"):
        return "valu"
    if op.startswith("ds_bpermute") or op.startswith("ds_permute") or op.startswith("ds_swizzle"):
        return "lds-xlane"
    if op.startswith("ds_read"):
        return "lds-read"
    if op.startswith("ds_write"):
        return "lds-write"
    if op.startswith("ds_"):
        return "lds-other"
    if op.startswith(("global_load", "buffer_load", "flat_load")):
        return "vmem-load"
    if op.startswith(("global_store", "buffer_store", "flat_store")):
        return "vmem-store"
    if op.startswith(("s_load", "s_buffer_load")):
        return "smem"
    if op.startswith("s_waitcnt"):
        return "waitcnt"
    if op.startswith("s_barrier"):
        return "barrier"
    if op.startswith("s_nop"):
        return "nop"
    if op == "OTHER_MSG":
        return "msg"
    return "salu"


def norm(asm):   # static identity without register numbers
    return re.sub(r"\b[vs]\[\d+:\d+\]|\b[vs]\d+\b", "r", asm).strip()


class Wave:
    def __init__(self, evs):
        evs.sort(key=lambda e: (e["ts"], e["args"].get("inst idx", 0)))
        self.ts = [e["ts"] for e in evs]
        self.clk = [e["args"].get("clks", 4) for e in evs]
        self.asm = [e["args"]["assembly"].strip() for e in evs]
        self.pc = [e["args"].get("offset", "?") for e in evs]
        self.op = [s.split()[0] for s in self.asm]
        self.cl = [cls(o) for o in self.op]
        n = len(self.ts)
        self.bars = [k for k in range(n) if self.op[k] == "s_barrier"]
        # static identity of an instruction = (position after the step's first barrier, assembly)
        self.key = [None] * n
        for j in range(0, len(self.bars) - 2, 2):
            for q in range(self.bars[j], self.bars[j + 2]):
                self.key[q] = f"{q - self.bars[j]:4d} {self.asm[q]}"
        # replay the in-order counters: producer of each binding waitcnt
        self.prod = {}
        vm, lg = [], []
        for k in range(n):
            c = self.cl[k]
            if c in ("vmem-load", "vmem-store"):
                vm.append(k)
            elif c.startswith("lds") or c == "smem":
                lg.append(k)
            elif c == "waitcnt":
                p = {}
                mv = re.search(r"vmcnt\((\d+)\)", self.asm[k])
                ml = re.search(r"lgkmcnt\((\d+)\)", self.asm[k])
                if mv:
                    N = int(mv.group(1))
                    if len(vm) > N:
                        p["vm"] = vm[len(vm) - N - 1]
                        vm = vm[len(vm) - N:] if N else []
                if ml:
                    N = int(ml.group(1))
                    if len(lg) > N:
                        p["lgkm"] = lg[len(lg) - N - 1]
                        lg = lg[len(lg) - N:] if N else []
                if p:
                    self.prod[k] = p
        # MMAC dependency: for each instruction, the latest MMAC whose destination it reads
        self.mdep = {}
        last = {}   # vreg -> mmac index
        for k in range(n):
            ops = self.asm[k][len(self.op[k]):]
            parts = ops.split(",")
            if self.cl[k] == "mmac":
                srcs = vregs(",".join(parts[1:]))
            elif self.op[k].startswith(("global_store", "ds_write", "buffer_store")):
                srcs = vregs(ops)
            else:
                srcs = vregs(",".join(parts[1:]))
            hit = [last[r] for r in srcs if r in last]
            if hit:
                self.mdep[k] = max(hit)
            if self.cl[k] == "mmac":
                for r in vregs(parts[0]):
                    last[r] = k
            elif parts and parts[0].strip():
                for r in vregs(parts[0]):
                    last.pop(r, None)


def lat_mem(cl):
    return 650 if cl.startswith("vmem") else 120


tot = collections.Counter()
by_pc = collections.Counter()
by_op = collections.Counter()   # (category, opcode) -> cycles
pc_asm = {}
nsteps_all = 0
step_cycles = []
files = sorted(glob.glob(f"{a.dir}/*thread_trace_{a.dispatch}__*.json"))
assert files, "no trace files"
for f in files:
    ev = json.load(open(f))["traceEvents"]
    st = collections.defaultdict(list)
    for e in ev:
        if e.get("ph") == "X" and "assembly" in e.get("args", {}):
            st[(e["pid"], e["tid"])].append(e)
    W = [Wave(v) for v in st.values()]
    nb = min(len(w.bars) for w in W)
    if nb < 40 or any(len(w.bars) != nb for w in W):
        print(f, "skip: barrier counts", [len(w.bars) for w in W])
        continue
    b0 = (int(0.15 * nb) // 2) * 2
    b1 = (int(0.85 * nb) // 2) * 2
    rel = lambda b: min(w.ts[w.bars[b]] for w in W)
    t_stop = rel(b0)
    # start: last arriving wave at b1
    def arrival(w, b):
        i = w.bars[b]
        return w.ts[i - 1] + w.clk[i - 1]
    wi = max(range(len(W)), key=lambda j: arrival(W[j], b1))
    w = W[wi]
    k = w.bars[b1]
    cnt = collections.Counter()
    cur_t = w.ts[k]
    tot_start = cur_t
    bidx = {j: {i: bb for bb, i in enumerate(W[j].bars)} for j in range(len(W))}
    while True:
        x, p = k, k - 1
        if w.ts[p] < t_stop:
            cnt["issue:" + w.cl[p]] += max(0, w.ts[x] - t_stop)
            break
        gap = w.ts[x] - w.ts[p] - w.clk[p]
        jumped = False
        if gap > a.thresh:
            if w.cl[x] == "barrier":
                b = bidx[wi][x]
                wl = max(range(len(W)), key=lambda j: arrival(W[j], b))
                if wl != wi:
                    t_arr = arrival(W[wl], b)
                    cnt["barrier:sync"] += w.ts[x] - t_arr
                    by_pc[(w.key[x], "barrier:sync")] += w.ts[x] - t_arr
                    wi, w = wl, W[wl]
                    k = w.bars[b]
                    # continue from the arrival point (the last instruction before its s_barrier)
                    cnt["issue:" + w.cl[k - 1]] += w.clk[k - 1]
                    by_pc[(w.key[k - 1], "issue:" + w.cl[k - 1])] += w.clk[k - 1]
                    k = k - 1
                    jumped = True
                else:
                    cnt["barrier:sync"] += gap
                    by_pc[(w.key[x], "barrier:sync")] += gap
            elif w.cl[x] == "waitcnt" and x in w.prod:
                pr = w.prod[x]
                kp = max(pr.values(), key=lambda q: w.ts[q] + lat_mem(w.cl[q]))
                if w.ts[kp] >= t_stop:
                    key = "lat:" + w.cl[kp]
                    cnt[key] += w.ts[x] - w.ts[kp]
                    by_pc[(w.key[kp], key)] += w.ts[x] - w.ts[kp]
                    pc_asm[w.key[kp]] = ""
                    k = kp
                    jumped = True
            elif x in w.mdep and w.ts[x] - w.ts[w.mdep[x]] <= a.mmac_window:
                km = w.mdep[x]
                if km < x - 1 or True:
                    key = "dep:mmac->" + w.cl[x]
                    cnt[key] += w.ts[x] - w.ts[km]
                    by_pc[(w.key[x], key)] += w.ts[x] - w.ts[km]
                    pc_asm[w.key[x]] = ""
                    k = km
                    jumped = True
            if not jumped and w.cl[x] not in ("barrier",):
                key = "stall:" + w.cl[x]
                cnt[key] += gap
                by_pc[(w.key[x], key)] += gap
                pc_asm[w.key[x]] = ""
        if not jumped:
            cnt["issue:" + w.cl[p]] += w.clk[p] + (0 if gap > a.thresh else max(0, gap))
            by_pc[(w.key[p], "issue:" + w.cl[p])] += w.clk[p] + (0 if gap > a.thresh else max(0, gap))
            pc_asm[w.key[p]] = ""
            k = p
        if k <= 0:
            break
    nst = (b1 - b0) // 2
    span = tot_start - t_stop
    acc = sum(cnt.values())
    step_cycles.append(span / nst)
    print(f"{f.split('/')[-1]}: {nst} steps, {span / nst:.0f} cycles/step (attributed {acc / nst:.0f})")
    for kk, v in cnt.items():
        tot[kk] += v / nst
    nsteps_all += 1
    for kk in list(by_pc):
        pass
for (kk, cat), v in by_pc.items():
    by_op[(cat, kk.split()[1] if kk and len(kk.split()) > 1 else "?")] += v
for kk in list(tot):
    tot[kk] /= nsteps_all
by_pc_step = {k: v / (nsteps_all * nst) for k, v in by_pc.items()}
S = sum(tot.values())
print(f"\n== critical path per step (mean of {nsteps_all} CTAs): {S:.0f} cycles")
grp = collections.Counter()
for kk, v in tot.items():
    grp[kk.split(":")[0]] += v
for g, v in grp.most_common():
    print(f"  {g:8s} {v:7.0f}  {100 * v / S:5.1f}%")
print()
for kk, v in tot.most_common():
    print(f"  {kk:28s} {v:7.0f}  {100 * v / S:5.1f}%")
print(f"\n== top {a.top} critical instructions (cycles/step)")
for (pc, key), v in sorted(by_pc_step.items(), key=lambda kv: -kv[1])[:a.top]:
    print(f"  {key:24s} {v:6.0f}  {str(pc)[:90]}")
print(f"\n== top opcodes on the critical path (cycles/step)")
for (cat, op), v in sorted(by_op.items(), key=lambda kv: -kv[1])[:a.top]:
    print(f"  {cat:24s} {op:28s} {v / (nsteps_all * nst):7.1f}")
if a.json:
    json.dump({"per_step_cycles": step_cycles, "categories": dict(tot),
               "top": [[pc, key, v, pc_asm.get(pc, "")] for (pc, key), v in
                       sorted(by_pc_step.items(), key=lambda kv: -kv[1])[:200]]}, open(a.json, "w"), indent=1)
