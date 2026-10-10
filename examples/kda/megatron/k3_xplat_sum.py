"""Summarize the A800 / BW K3-slice Megatron matrices (k3_matrix_nv.sh, k3_matrix_bw52.sh).
Per (platform, arm, recompute, seq, tp): step time (median of iters 4-9), lm loss trace, K3_MEM
peaks, and from the rank0 profiler trace of steps 10-11: total GPU kernel time and the KDA kernel
time (FLA Triton kernels or hip_kda G2 kernels), per step.
    python3 k3_xplat_sum.py /public/home/tanbo/xplat/megatron nvl4 bwl4 > sum.json"""
import collections
import glob
import gzip
import json
import os
import re
import statistics as st
import sys

OUT, TAGS = sys.argv[1], sys.argv[2:]


def kcls(n):
    l = n.lower()
    if "g2::" in l or l.startswith("g2_") or "p3_wy_s4" in l or "_zn2g2" in l:
        return "kda"
    if any(k in l for k in ("chunk_kda", "chunk_gla", "chunk_gated_delta_rule", "kda_gate", "l2norm_fwd_kernel",
                            "l2norm_bwd_kernel", "recompute_w_u", "chunk_local_cumsum", "fused_beta", "kda_")):
        return "kda"
    if "nccl" in l or "rccl" in l or "allreduce" in l or "all_gather" in l:
        return "comm"
    return "other"


def trace_ms(d):
    """KDA / total kernel ms per profiled step from the newest trace in the run dir."""
    fs = sorted(glob.glob(f"{d}/**/*.json*", recursive=True), key=os.path.getmtime)
    fs = [f for f in fs if "trace" in os.path.basename(f) or f.endswith(".pt.trace.json") or f.endswith(".json.gz")]
    if not fs:
        return None
    f = fs[-1]
    raw = gzip.open(f).read() if f.endswith(".gz") else open(f, "rb").read()
    ev = json.loads(raw).get("traceEvents", [])
    per = collections.Counter()
    for e in ev:
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_op"):
            per[kcls(e["name"])] += e.get("dur", 0)
    steps = 2   # profile-step-start 10 .. end 11 -> steps 10 and 11 (check: one trace per run)
    tot = sum(per.values())
    return {"trace": os.path.relpath(f, OUT), "gpu_ms_per_step": round(tot / 1000 / steps, 2),
            "kda_ms_per_step": round(per["kda"] / 1000 / steps, 2), "comm_ms_per_step": round(per["comm"] / 1000 / steps, 2),
            "kda_share_pct": round(100 * per["kda"] / tot, 2) if tot else None}


rows = []
for tag in TAGS:
    for log in sorted(glob.glob(f"{OUT}/logs/{tag}_*.log")):
        m = re.search(rf"{tag}_(\w+?)_r(\d)_s(\d+)_tp(\d+)\.log$", log)
        if not m:
            continue
        arm, rc, seq, tp = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        L = open(log, errors="ignore").read()
        r = {"tag": tag, "arm": arm, "recompute": rc, "seq": seq, "tp": tp, "heads_per_card": 96 // tp,
             "oom": ("OutOfMemoryError" in L or "out of memory" in L) and "iteration       12/" not in L}
        its = [(int(a), float(b), float(c)) for a, b, c in re.findall(
            r"iteration\s+(\d+)/\s*\d+ \|.*?elapsed time per iteration \(ms\): ([\d.]+).*?lm loss: ([\d.E+-]+)", L)]
        ts = [t for i, t, _ in its if 4 <= i <= 9]
        if ts:
            r["step_ms"] = round(st.median(ts), 1)
            r["loss"] = [round(x, 4) for _, _, x in its]
        mem = [tuple(map(float, g)) for g in re.findall(
            r"K3_MEM rank=0 it=(\d+) static_MB=([\d.]+) peak_MB=([\d.]+)", L)]
        if mem:
            r["peak_MB_rank0"] = max(p for it, s, p in mem if it >= 3) if any(it >= 3 for it, _, _ in mem) else None
        t = trace_ms(f"{OUT}/prof/{tag}_{arm}_r{rc}_s{seq}_tp{tp}")
        if t:
            r.update(t)
        rows.append(r)
print(json.dumps(rows, indent=1))
