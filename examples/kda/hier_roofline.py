#!/usr/bin/env python3
"""Hierarchical roofline for gfx936 (BW200/BW1000) from the kra machine calibration,
with the KDA G2 kernels placed by their measured HBM traffic and executed MMAC FLOPs.

Ceilings  : machines/gfx936-bw1000.json  (attainable = best measured config; theoretical where known)
Operator  : results/kda-gfx936-20261010-ceiling/g2_8192_{48,12}_prod.waterlevel.json
            point = MODEL FLOPs of the chunked KDA algorithm (compute.model_GFLOP, exact)
                    / measured HBM bytes (TCC_EA, summed over kernels) / summed kernel time.
            The kra "executed MMAC" number is an ESTIMATE from VALU issue slots (slots - insts) x 8192;
            transcendentals (4 slots) and fp32 MMAC (4096 FLOP) inflate it, so it is drawn only as an
            upper-bound marker, and per-kernel FLOP points are NOT plotted.
Kernels   : second panel = directly measured per-kernel utilizations (HBM, VMEM issue, VALU pipe).
Usage: python hier_roofline.py [out_dir]   (needs matplotlib)
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MACHINE = os.path.join(ROOT, "machines", "gfx936-bw1000.json")
WL = os.path.join(ROOT, "results", "kda-gfx936-20261010-ceiling", "g2_8192_{h}_prod.waterlevel.json")


def load():
    m = json.load(open(MACHINE))
    pk = m["peaks"]
    mem = [  # name, GB/s, style
        ("LDS", pk["lds_read"]["value"], "-"),
        ("L2", pk["l2_read"]["value"], "-"),
        ("HBM", pk["hbm_read"]["value"], "-"),
        ("HBM theoretical", pk["hbm_read"]["theoretical"], ":"),
    ]
    comp = [  # name, TFLOPS
        ("bf16 MMAC", pk["mmac_bf16"]["value"]),
        ("tf32 MMAC", pk["mmac_tf32"]["value"]),
        ("fp32 MMAC (16x16x8)", pk["mmac_f32x8"]["value"]),
        ("fp32 VALU FMA", pk["valu_fma_f32"]["value"]),
    ]
    kernels = {}
    for h in (48, 12):
        d = json.load(open(WL.format(h=h)))
        rows = []
        for r in d["rows"]:
            for k in r["kernels"]:
                if k["t_us"] < 20:                         # torch elementwise / reduce tails
                    continue
                rows.append(dict(name=k["name"].replace("_kernel", "").replace("g2_", ""),
                                 t_us=k["t_us"], bytes=(k["hbm_rd_MB"] + k["hbm_wr_MB"]) * 1e6,
                                 hbm=k["sol_mem_active"], vmem=k["sol_vmem_active"] / 0.92,
                                 pipe=k["sol_pipe_active"], cls=k["classification"]["class"]))
        allk = [k for r in d["rows"] for k in r["kernels"]]
        t = sum(k["t_us"] for k in allk) * 1e-6
        by = sum(k["hbm_rd_MB"] + k["hbm_wr_MB"] for k in allk) * 1e6
        c = d["compute"]
        op = dict(t_s=t, bytes=by, model=c["model_GFLOP"] * 1e9, executed=c["executed_mmac_GFLOP"] * 1e9)
        kernels[h] = (d.get("operator", ""), rows, op)
    return m, mem, comp, kernels


def plot(mem, comp, kernels, out):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(17, 7.5), gridspec_kw={"width_ratios": [1.35, 1]})
    xs = [10 ** (e / 20) for e in range(-40, 101)]          # 0.01 .. 1e5 FLOP/B
    colors = {"LDS": "tab:purple", "L2": "tab:blue", "HBM": "tab:red", "HBM theoretical": "tab:red"}
    top = max(c[1] for c in comp)
    for name, gbps, ls in mem:
        ys = [min(gbps * x / 1e3, top) for x in xs]
        ax.plot(xs, ys, ls, color=colors[name], lw=2 if ls == "-" else 1.2,
                label=f"{name} {gbps / 1e3:.2f} TB/s")
    ccol = {"bf16 MMAC": "k", "tf32 MMAC": "0.35", "fp32 MMAC (16x16x8)": "tab:green", "fp32 VALU FMA": "tab:orange"}
    for name, tf in comp:
        ax.axhline(tf, color=ccol[name], ls="--", lw=1.3)
        ax.text(1.5e4, tf * 1.07, f"{name} {tf:.1f} TFLOPS", ha="right", fontsize=9, color=ccol[name])
    for name, gbps, ls in mem:                                # ridge points
        if ls != "-":
            continue
        for cname, tf in (comp[0], comp[2]):
            r = tf * 1e3 / gbps
            ax.plot([r], [tf], "o", color=colors[name], ms=5)
            ax.annotate(f"{r:.0f}", (r, tf), textcoords="offset points", xytext=(4, -12),
                        fontsize=8, color=colors[name])
    mk = {48: ("s", "tab:red"), 12: ("^", "tab:blue")}
    for h, (_, rows, op) in kernels.items():
        m_, c_ = mk[h]
        x1, y1 = op["model"] / op["bytes"], op["model"] / op["t_s"] / 1e12
        x2, y2 = op["executed"] / op["bytes"], op["executed"] / op["t_s"] / 1e12
        ax.plot([x1], [y1], m_, color=c_, ms=9, label=f"KDA fwd+bwd 8K/H{h}: model FLOPs (exact)")
        ax.plot([x2], [y2], m_, color=c_, ms=9, mfc="none", label=f"8K/H{h}: executed-MMAC estimate (upper bound)")
        ax.plot([x1, x2], [y1, y2], ":", color=c_, lw=1)
        ax.annotate(f"H{h}: AI {x1:.0f}, {y1:.1f} TFLOPS", (x1, y1), textcoords="offset points",
                    xytext=(-10, -16 if h == 48 else 10), fontsize=8, color=c_, ha="right")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.1, 2e4); ax.set_ylim(0.5, 900)
    ax.set_xlabel("Arithmetic intensity (FLOP / HBM byte)")
    ax.set_ylabel("Performance (TFLOPS)")
    ax.set_title("gfx936 (BW200/BW1000) hierarchical roofline (measured attainable ceilings)\n"
                 "dots = ridge points (FLOP/B) of bf16 MMAC and fp32 MMAC vs LDS / L2 / HBM")
    ax.grid(True, which="both", ls=":", lw=0.5)
    ax.legend(loc="lower right", fontsize=8)
    # panel 2: measured per-kernel utilization, 8K/H48 (sorted by time)
    _, rows, _ = kernels[48]
    rows = sorted(rows, key=lambda r: -r["t_us"])
    ys = range(len(rows))
    w = 0.27
    bx.barh([y - w for y in ys], [r["hbm"] for r in rows], w, color="tab:red", label="HBM BW / attainable")
    bx.barh(list(ys), [r["vmem"] for r in rows], w, color="tab:olive", label="VMEM issue / saturation (0.92)")
    bx.barh([y + w for y in ys], [r["pipe"] for r in rows], w, color="tab:orange", label="VALU pipe slots")
    bx.set_yticks(list(ys)); bx.set_yticklabels([f"{r['name']} ({r['t_us']:.0f} us)" for r in rows], fontsize=8)
    bx.invert_yaxis(); bx.set_xlim(0, 1.05); bx.axvline(0.8, color="0.5", ls=":", lw=1)
    bx.set_xlabel("utilization of the attainable peak (per active CU)")
    bx.set_title("KDA G2 kernels, 8K/H48: which roof binds (measured PMC)")
    bx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, fontsize=8); bx.grid(True, axis="x", ls=":", lw=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def table(m, mem, comp, kernels, out):
    L = ["# gfx936 分层 roofline（kra 标定 + KDA G2 kernel 落点）", "",
         f"标定：`machines/gfx936-bw1000.json`（{m['provenance']['host']}，{m['provenance']['timestamp_utc']}）。",
         "带宽与算力都取**实测可达值**（最佳配置），理论值单列。", "",
         "## 屋顶线", "", "| 层级 | 可达带宽 | 理论带宽 |", "|---|---|---|"]
    for name, gbps, ls in mem:
        if ls == "-":
            th = m["peaks"]["hbm_read"]["theoretical"] if name == "HBM" else None
            L.append(f"| {name} | {gbps / 1e3:.2f} TB/s | {f'{th / 1e3:.2f} TB/s' if th else '—'} |")
    L += ["", "| 计算 | 可达算力 |", "|---|---|"] + [f"| {n} | {t:.1f} TFLOPS |" for n, t in comp]
    L += ["", "## 拐点（FLOP/Byte = 可达算力 ÷ 可达带宽）", "",
          "| 计算 \\ 存储层 | LDS | L2 | HBM | HBM（理论带宽） |", "|---|---|---|---|---|"]
    for n, t in comp:
        L.append(f"| {n} | " + " | ".join(f"{t * 1e3 / g:.1f}" for _, g, _ in mem) + " |")
    L += ["", "## KDA 算子（fwd+bwd）在 HBM roofline 上的位置", "",
          "| 形态 | 模型 FLOPs | HBM 流量 | kernel 时间 | 算术强度（模型 FLOPs） | 实际算力 | 实际 HBM 带宽 | 执行 MMAC 估计（上界） |",
          "|---|---|---|---|---|---|---|---|"]
    for h, (opname, rows, op) in kernels.items():
        L.append(f"| 8K/H{h} | {op['model'] / 1e9:.1f} GFLOP | {op['bytes'] / 1e9:.2f} GB | {op['t_s'] * 1e3:.2f} ms | "
                 f"{op['model'] / op['bytes']:.1f} FLOP/B | {op['model'] / op['t_s'] / 1e12:.1f} TFLOPS | "
                 f"{op['bytes'] / op['t_s'] / 1e9:.0f} GB/s | AI {op['executed'] / op['bytes']:.0f}，{op['executed'] / op['t_s'] / 1e12:.1f} TFLOPS |")
    L += ["", f"来源：{kernels[48][0]}；FLOPs 为分块 KDA 的模型 FLOPs（反向 = 2 × 前向）。", "",
          "## 每个 kernel 的实测利用率（8K/H48，按时间排序）", "",
          "| kernel | 时间 µs | HBM 流量 MB | HBM 带宽占可达 | VMEM 发射占饱和 | VALU 发射槽 | kra 分类 |",
          "|---|---|---|---|---|---|---|"]
    for r in sorted(kernels[48][1], key=lambda r: -r["t_us"]):
        L.append(f"| {r['name']} | {r['t_us']:.0f} | {r['bytes'] / 1e6:.0f} | {r['hbm']:.0%} | {r['vmem']:.0%} | {r['pipe']:.0%} | {r['cls']} |")
    open(out, "w").write("\n".join(L) + "\n")


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "gfx936-hier-roofline-20261013")
    os.makedirs(out_dir, exist_ok=True)
    m, mem, comp, kernels = load()
    plot(mem, comp, kernels, os.path.join(out_dir, "gfx936_hier_roofline.png"))
    table(m, mem, comp, kernels, os.path.join(out_dir, "README.md"))
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
