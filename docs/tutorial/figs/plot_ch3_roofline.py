"""第 3 章插图：BW1000（gfx936）Roofline 与算子落点。

数据来源：
  - 屋顶与斜线：machines/gfx936-bw1000.json（bw7 标定，2026-10-10）
  - GEMM：peak_flops_bench bw52 hipBLASLt bf16 8192^3 TN 最佳 331.6 TFLOPS（2026-10-30）
  - KDA fwd+bwd 16K/TP8（H=12，P-88）：99.1 GFLOP、实测 DRAM 4.722 GB、5.32 ms；算法本身强度 122 FLOP/B
  - l2n_apply（KDA 8K/H12）：HBM 1297 GB/s；FLOPs 按每元素 1 次乘、读写各 2 B 估算（约 0.25 FLOP/B）
用法：python3 plot_ch3_roofline.py  -> fig3_1_bw1000_roofline.png, fig3_2_bw1000_operators.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

HERE = os.path.dirname(os.path.abspath(__file__))
M = json.load(open(os.path.join(HERE, "../../../machines/gfx936-bw1000.json")))
pk = M["peaks"]
BW = pk["hbm_read"]["value"] / 1e3          # TB/s，实测可达
BW_TH = 1.843                               # TB/s，理论（4096 bit x 1.8 GHz x 2 / 8）
ROOFS = [  # (名称, TFLOPS, 颜色)
    ("bf16 MMAC", pk["mmac_bf16"]["value"], "#d62728"),
    ("tf32 MMAC", pk["mmac_tf32"]["value"], "#9467bd"),
    ("fp32 MMAC（16x16x8）", pk["mmac_f32x8"]["value"], "#2ca02c"),
    ("fp32 VALU（打包 FMA）", pk["valu_fma_f32"]["value"], "#ff7f0e"),
]
X = np.logspace(-1, 4, 400)


def base(ax, title, faint=False):
    a = 0.35 if faint else 1.0
    ax.loglog(X, BW * X, color="black", lw=2.2, alpha=a, label=f"HBM 实测可达 {BW:.2f} TB/s")
    ax.loglog(X, BW_TH * X, color="black", lw=1.2, ls=":", alpha=a, label=f"HBM 理论 {BW_TH:.2f} TB/s")
    for name, p, c in ROOFS:
        ridge = p / BW
        xs = X[X >= ridge]
        ax.loglog(xs, np.full_like(xs, p), color=c, lw=2, alpha=a)
        ax.loglog([ridge], [p], "o", color=c, ms=6, alpha=a)
        if not faint:
            ax.annotate(f"拐点 {ridge:.0f}" if ridge >= 100 else f"拐点 {ridge:.1f}", (ridge, p), textcoords="offset points", xytext=(6, -14),
                        color=c, fontsize=9)
        ax.text(9e3, p * 1.06, f"{name} {p:.1f} TFLOPS", ha="right", va="bottom", color=c, fontsize=9, alpha=a)
    ax.set_xlim(0.1, 1e4)
    ax.set_ylim(0.1, 900)
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("计算强度 I（FLOP / HBM 字节）")
    ax.set_ylabel("性能（TFLOPS）")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)


# ---------- 图 3-1 ----------
fig, ax = plt.subplots(figsize=(10, 6.4), dpi=150)
base(ax, "图 3-1　BW1000（gfx936，80 CU，1500 MHz）Roofline：实测可达屋顶")
r = ROOFS[0][1] / BW
ax.axvspan(0.1, r, color="#1f77b4", alpha=0.05)
ax.axvspan(r, 1e4, color="#d62728", alpha=0.04)
ax.text(0.15, 300, "左侧：访存受限区\n（落在斜线下，看 MBU）", fontsize=10, color="#1f77b4")
ax.text(700, 1.0, "右侧：算力受限区\n（落在平顶下，看 MFU）\n以 bf16 MMAC 为例", fontsize=10, color="#d62728")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig3_1_bw1000_roofline.png"))

# ---------- 图 3-2 ----------
fig, ax = plt.subplots(figsize=(10, 6.4), dpi=150)
base(ax, "图 3-2　三个算子在 BW1000 Roofline 上的位置", faint=True)
pts = [
    # (I, TFLOPS, 标签, 颜色, marker, filled)
    (8192 / 3, 331.6, "GEMM bf16 M=N=K=8192（hipBLASLt）\nI=2731，331.6 TFLOPS，MFU 70%", "#d62728", "s", True),
    (21.0, 99.1 / 5.32, "KDA 前向+反向 16K/TP8（实现）\nI=21，18.6 TFLOPS，MBU 66%", "#1f77b4", "o", True),
    (0.25, 0.25 * 1.297, "l2n_apply（KDA 8K/H12）\nI≈0.25（估算），HBM 1297 GB/s，MBU 97%", "#2ca02c", "^", True),
]
for i, p, lab, c, mk, fill in pts:
    ax.plot([i], [p], mk, color=c, ms=10, mfc=c if fill else "none", mew=2)
offs = {0: (-215, -42), 1: (-40, -62), 2: (14, -6)}
for k, (i, p, lab, c, *_r) in enumerate(pts):
    if k == 0:
        ax.annotate(lab, (i, p), xytext=(600, 8), textcoords="data", fontsize=9, color=c,
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.2))
    else:
        ax.annotate(lab, (i, p), textcoords="offset points", xytext=offs[k], fontsize=9, color=c)
ai, aa = 122.0, 122.0 * BW
ax.plot([ai], [aa], "o", color="#1f77b4", ms=10, mfc="none", mew=2)
ax.annotate("", xy=(21.0, 99.1 / 5.32), xytext=(ai, aa),
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.6, ls="--"))
ax.annotate("KDA 算法本身强度 I=122\n（完全融合时的上限 163 TFLOPS，\n仍在拐点 355 左侧）", (ai, aa),
            textcoords="offset points", xytext=(-200, 4), fontsize=9, color="#1f77b4")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig3_2_bw1000_operators.png"))
print("ok", BW, [(n, round(p / BW, 1)) for n, p, _ in ROOFS])
