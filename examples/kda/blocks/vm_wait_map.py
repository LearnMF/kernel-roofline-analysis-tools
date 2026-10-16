"""Which vmcnt waits pay for which VMEM instruction (per wave, steady-state steps of a recurrence dispatch).
Replays each wave's in-order VMEM queue (loads AND stores: gfx9 vmcnt counts both) and, at every
s_waitcnt vmcnt(N) whose release gap exceeds --min, charges the gap to the instruction that had to retire
(the oldest outstanding beyond N).  Positions are instruction offsets from the step's first barrier.
    python3 vm_wait_map.py <sqtt dir> --dispatch 3 [--min 40] [--top 20]"""
import argparse
import collections
import glob
import json
import re

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--dispatch", default="3")
ap.add_argument("--min", type=int, default=40)
ap.add_argument("--top", type=int, default=20)
ap.add_argument("--bars-per-step", type=int, default=2)
a = ap.parse_args()
VM = re.compile(r"^(global_|buffer_|flat_)(load|store|atomic)")
WC = re.compile(r"vmcnt\((\d+)\)")
agg = collections.Counter()
cnt = collections.Counter()
nsteps = 0
for f in sorted(glob.glob(f"{a.dir}/*thread_trace_{a.dispatch}__*.json")):
    ev = json.load(open(f))["traceEvents"]
    waves = collections.defaultdict(list)
    for e in ev:
        if e.get("ph") == "X" and "assembly" in e.get("args", {}):
            waves[(e["pid"], e["tid"])].append(e)
    for w, s in waves.items():
        s.sort(key=lambda e: (e["ts"], e["args"]["inst idx"]))
        bars = [k for k, e in enumerate(s) if e["name"] == "s_barrier"]
        if len(bars) < 40:
            continue
        starts = bars[10:-10:a.bars_per_step]
        q = []                                   # outstanding VMEM: (pos-in-step, asm)
        step_of = {}
        for b0, b1 in zip(starts[:-1], starts[1:]):
            nsteps += 1
            for k in range(b0, b1):
                e, p = s[k], s[k - 1]
                asm = e["args"]["assembly"]
                pos = k - b0
                if VM.match(asm):
                    q.append((pos, asm.split()[0]))
                m = WC.search(asm) if asm.startswith("s_waitcnt") else None
                if m:
                    n = int(m.group(1))
                    gap = e["ts"] - p["ts"] - p["args"]["clks"]
                    if len(q) > n:
                        culprit = q[len(q) - n - 1]
                        if gap > a.min:
                            agg[(culprit[0], culprit[1], pos, asm[:40])] += gap
                            cnt[(culprit[0], culprit[1], pos, asm[:40])] += 1
                        q = q[len(q) - n:]
print(f"{nsteps} wave-steps; vmcnt waits > {a.min} cycles charged to the instruction that had to retire")
print("cycles/wave-step  count  culprit(pos opcode)  <-  wait(pos asm)")
for k_, v in sorted(agg.items(), key=lambda kv: -kv[1])[:a.top]:
    print(f"{v / max(1, nsteps):8.1f}  {cnt[k_]:6d}  @{k_[0]:4d} {k_[1]:24s} <- @{k_[2]:4d} {k_[3]}")
