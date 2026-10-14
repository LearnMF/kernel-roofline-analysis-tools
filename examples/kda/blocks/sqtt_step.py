"""Ground truth for the ISA simulator: per-step time and per-step stall breakdown of a recurrence kernel from hipprof
SQTT (--sqtt --sqtt-type 1 --output-type 0).  A step = two s_barrier records of a wave; the steady state skips the
first and last 15% of each wave's steps.  Gaps (cycles between the previous instruction's issue + clks and this
issue) are charged like sqtt_hotspots.py: at s_waitcnt -> waitcnt (split vm / lgkm by the counters in the text),
at s_barrier -> barrier, otherwise -> issue delay of that instruction's class.
    python3 sqtt_step.py <dir> --dispatch 0 [--barriers-per-step 2]"""
import argparse
import collections
import glob
import json
import re

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--dispatch", default="0")
ap.add_argument("--barriers-per-step", type=int, default=2)
a = ap.parse_args()


def cls(asm):
    op = asm.split()[0]
    if op.startswith("v_mmac"):
        return "mmac"
    if op.startswith(("v_exp", "v_rcp", "v_log", "v_sqrt", "v_rsq")):
        return "trans"
    if op.startswith("v_pk"):
        return "pk"
    if op.startswith("v_"):
        return "valu"
    if op.startswith("ds_"):
        return "lds"
    if op.startswith(("global_", "buffer_", "flat_")):
        return "vmem"
    return "salu"


steps, stall, per_wave = [], collections.Counter(), []
files = sorted(glob.glob(f"{a.dir}/*thread_trace_{a.dispatch}__*.json"))
assert files, "no trace files"
for f in files:
    ev = json.load(open(f))["traceEvents"]
    streams = collections.defaultdict(list)
    for e in ev:
        if e.get("ph") == "X" and "assembly" in e.get("args", {}):
            streams[(e["pid"], e["tid"])].append(e)
    for s in streams.values():
        s.sort(key=lambda e: (e["ts"], e["args"].get("inst idx", 0)))
        bars = [k for k, e in enumerate(s) if e["name"] == "s_barrier"]
        nst = len(bars) // a.barriers_per_step
        if nst < 20:
            continue
        b0 = bars[int(0.15 * nst) * a.barriers_per_step]
        b1 = bars[int(0.85 * nst) * a.barriers_per_step]
        n = (int(0.85 * nst) - int(0.15 * nst))
        span = s[b1]["ts"] - s[b0]["ts"]
        per_wave.append(span / n)
        for k in range(b0 + 1, b1 + 1):
            x, p = s[k], s[k - 1]
            gap = max(0, x["ts"] - p["ts"] - p["args"].get("clks", 4))
            asm = x["args"]["assembly"]
            if x["name"] == "s_waitcnt":
                key = "wait:" + ("+".join(m for m in ("vmcnt", "lgkmcnt") if m in asm))
            elif x["name"] == "s_barrier":
                key = "barrier"
            else:
                key = "issue:" + cls(asm)
            stall[key] += gap / n
            stall["busy:" + cls(asm)] += p["args"].get("clks", 4) / n
per_wave.sort()
print(f"{len(per_wave)} waves; cycles per step: median {per_wave[len(per_wave) // 2]:.0f}  "
      f"min {per_wave[0]:.0f}  max {per_wave[-1]:.0f}  ({per_wave[len(per_wave) // 2] / 1500:.3f} us @1.5 GHz)")
tw = len(per_wave)
print("per wave per step (cycles):", {k: round(v / tw) for k, v in sorted(stall.items(), key=lambda kv: -kv[1])})
