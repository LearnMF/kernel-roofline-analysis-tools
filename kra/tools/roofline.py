"""通用 Roofline / 分层 Roofline 画图与落点计算。

输入：标定文件（python -m kra calibrate 生成的 machines/<卡>.json）和一个落点 CSV，列为
    name, flops, time_us [, hbm_bytes] [, l2_bytes] [, lds_bytes] [, roof]
roof 取 mmac_bf16 / mmac_f16 / mmac_tf32 / mmac_f32x8 / valu_fma_f32（缺省 mmac_bf16），用于计算 MFU。
只给 hbm_bytes 画单层 Roofline；同时给 l2_bytes、lds_bytes 时，每个 kernel 画三个点（分层 Roofline）。

    python -m kra.tools.roofline --machine machines/gfx936-bw1000.json --points pts.csv --out roof.png [--title "..."]

同时在标准输出打印每个点的计算强度、实际性能、各层利用率、MFU，以及到上方屋顶的比例。
"""
import argparse
import csv
import json
import sys

ROOFS = ["mmac_bf16", "mmac_tf32", "mmac_f32x8", "valu_fma_f32"]
LEVELS = [("hbm", "hbm_read", "HBM", "o"), ("l2", "l2_read", "L2", "v"), ("lds", "lds_read", "LDS", "D")]


def peaks(machine):
    pk = machine["peaks"]
    return ({k: pk[k]["value"] * 1e12 for k in pk if k.startswith(("mmac_", "valu_")) and pk[k].get("value")},
            {lv: pk[key]["value"] * 1e9 for lv, key, _, _ in LEVELS if pk.get(key, {}).get("value")})


def analyze(machine, rows):
    comp, bw = peaks(machine)
    out = []
    for r in rows:
        f, t = float(r["flops"]), float(r["time_us"]) * 1e-6
        roof = r.get("roof") or "mmac_bf16"
        perf = f / t
        res = {"name": r["name"], "TFLOPS": perf / 1e12, "MFU": perf / comp[roof], "roof": roof}
        for lv, _, label, _ in LEVELS:
            b = r.get(f"{lv}_bytes")
            if b not in (None, ""):
                b = float(b)
                res[f"I_{label}"] = f / b if b else float("inf")
                res[f"u_{label}"] = b / t / bw[lv]
        if "I_HBM" in res:
            res["to_roof"] = perf / min(comp[roof], bw["hbm"] * res["I_HBM"])
        out.append(res)
    return out


def plot(machine, rows, res, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    comp, bw = peaks(machine)
    fig, ax = plt.subplots(figsize=(10, 6.4), dpi=150)
    x = np.logspace(-2, 4, 400)
    hier = any(f"I_L2" in r for r in res)
    colors = {"hbm": "#d62728", "l2": "#1f77b4", "lds": "#9467bd"}
    for lv, _, label, _ in (LEVELS if hier else LEVELS[:1]):
        ax.loglog(x, bw[lv] * x / 1e12, color=colors[lv], lw=2, label=f"{label} {bw[lv] / 1e12:.2f} TB/s")
    for k in ROOFS:
        if k in comp:
            ax.axhline(comp[k] / 1e12, ls="--", lw=1, color="gray")
            ax.text(9e3, comp[k] / 1e12 * 1.05, f"{k} {comp[k] / 1e12:.1f} TFLOPS", ha="right", fontsize=8, color="gray")
    for r in res:
        ys = r["TFLOPS"]
        pts = [(r[f"I_{label}"], mk, colors[lv]) for lv, _, label, mk in LEVELS if f"I_{label}" in r]
        for xi, mk, c in pts:
            ax.loglog([xi], [ys], mk, color=c, ms=7)
        if len(pts) > 1:
            ax.plot([min(p[0] for p in pts), max(p[0] for p in pts)], [ys, ys], color="gray", lw=0.6)
        ax.annotate(r["name"], (pts[0][0], ys), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xlabel("计算强度 I（FLOP / 该层字节）"); ax.set_ylabel("性能（TFLOPS）")
    ax.set_title(title or ("分层 Roofline" if hier else "Roofline") + f"：{machine.get('id', '')}")
    ax.grid(True, which="both", alpha=0.25); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(path)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m kra.tools.roofline", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--machine", required=True); p.add_argument("--points", required=True)
    p.add_argument("--out"); p.add_argument("--title")
    a = p.parse_args(argv)
    machine = json.load(open(a.machine))
    rows = list(csv.DictReader(open(a.points)))
    res = analyze(machine, rows)
    for r in res:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, ensure_ascii=False))
    if a.out:
        plot(machine, rows, res, a.out, a.title)
    return 0


if __name__ == "__main__":
    sys.exit(main())
