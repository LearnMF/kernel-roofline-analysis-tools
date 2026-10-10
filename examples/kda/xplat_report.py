"""Cross-platform KDA report: hip_kda G2 on BW (gfx936) vs FLA chunk_kda (Triton) on A800.

    python3 xplat_report.py --nv seq_nv210.jsonl [--nv2 seq_nv211.jsonl] --bw seq/bw.jsonl \
        --bwdev bw_dev.json [--nvpeak nv_peaks.json]

Ratios are written as "the other side's time / ours" (>1 = BW-G2 faster), per the team rule.
Device time: A800 = torch.profiler (CUPTI) kernel sum of one fwd+bwd; BW = hipprof trace kernel sum
(torch.profiler over-reports on DCU at large shapes).  Wall: CUDA/HIP events around one fwd+bwd
(includes host launch overhead).  ckpt: one Megatron reentrant-checkpoint step, wall."""
import argparse
import collections
import json
import statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("--nv", required=True)
ap.add_argument("--nv2", default=None)
ap.add_argument("--bw", required=True)
ap.add_argument("--bwdev", required=True)
ap.add_argument("--bw-hbm", type=float, default=1339.2, help="BW attainable HBM read GB/s (machine.json)")
ap.add_argument("--nv-hbm", type=float, default=1771.2, help="A800 attainable HBM read GB/s (nv_peaks.py)")
a = ap.parse_args()


def load(p):
    agg, env = collections.defaultdict(lambda: collections.defaultdict(list)), None
    for line in open(p):
        if line.strip():
            d = json.loads(line)["data"]
            env = d.get("env", env)
            for arm, v in d["res"].items():
                for m, x in v.items():
                    if isinstance(x, (int, float)):
                        agg[(d["T"], d["H"])][(arm, m)].append(x)
    return agg, env


nv, env = load(a.nv)
nv2 = load(a.nv2)[0] if a.nv2 else None
bw, _ = load(a.bw)
dev = json.load(open(a.bwdev))
med = lambda A, k, arm, m: st.median(A[k][(arm, m)]) if A[k][(arm, m)] else float("nan")
shapes = sorted(set(nv) & set(bw))
L = [f"A800 stack: {env}", "",
     "## Device time (GPU only), fwd+bwd, ms", "",
     "| shape (T x H) | A800 FLA | A800 FLA nr | BW G2 | BW FLA | **A800 FLA / BW G2** | A800 FLA nr / BW G2 | BW FLA / A800 FLA | BW-normalized A800 FLA / BW G2 |",
     "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
bwr = a.nv_hbm / a.bw_hbm
for k in shapes:
    key = f"{k[0]}x{k[1]}"
    af, an = med(nv, k, "fla", "kernel_fb_ms"), med(nv, k, "fla_nr", "kernel_fb_ms")
    g, f = dev[key]["g2"]["device_fb_ms"], dev[key]["fla"]["device_fb_ms"]
    L.append(f"| {k[0]} x {k[1]} | {af:.2f} | {an:.2f} | {g:.2f} | {f:.2f} | **{af / g:.2f}** | {an / g:.2f} | {f / af:.2f} | {af / g * bwr:.2f} |")
L += ["", "## Wall time (one isolated call, host launch overhead included), ms", "",
      "| shape | A800 FLA fwd+bwd | BW G2 fwd+bwd | **A800 FLA / BW G2** | A800 FLA ckpt step | BW G2 ckpt step | **ckpt ratio** | A800 FLA nr / BW G2 |",
      "|---|---:|---:|---:|---:|---:|---:|---:|"]
for k in shapes:
    af, g = med(nv, k, "fla", "fb_ms"), med(bw, k, "g2", "fb_ms")
    ac, gc = med(nv, k, "fla", "ckpt_step_ms"), med(bw, k, "g2", "ckpt_step_ms")
    an = med(nv, k, "fla_nr", "fb_ms")
    L.append(f"| {k[0]} x {k[1]} | {af:.2f} | {g:.2f} | **{af / g:.2f}** | {ac:.2f} | {gc:.2f} | **{ac / gc:.2f}** | {an / g:.2f} |")
if nv2:
    L += ["", "## Cross-check: A800 with torch 2.11 (same triton 3.6.0 / FLA 0.5.1), device ms", "",
          "| shape | torch 2.10 | torch 2.11 | 2.11 / 2.10 |", "|---|---:|---:|---:|"]
    for k in shapes:
        x, y = med(nv, k, "fla", "kernel_fb_ms"), med(nv2, k, "fla", "kernel_fb_ms")
        L.append(f"| {k[0]} x {k[1]} | {x:.2f} | {y:.2f} | {y / x:.3f} |")
L += ["", "## Measurement spread (max-min)/median over processes", "",
      "| shape | A800 FLA device | BW G2 device | A800 FLA wall | BW G2 wall |", "|---|---:|---:|---:|---:|"]
sp = lambda A, k, arm, m: (max(A[k][(arm, m)]) - min(A[k][(arm, m)])) / st.median(A[k][(arm, m)])
for k in shapes:
    key = f"{k[0]}x{k[1]}"
    L.append(f"| {k[0]} x {k[1]} | {sp(nv, k, 'fla', 'kernel_fb_ms'):.1%} | {dev[key]['g2']['spread']:.1%} | "
             f"{sp(nv, k, 'fla', 'fb_ms'):.1%} | {sp(bw, k, 'g2', 'fb_ms'):.1%} |")
print("\n".join(L))
