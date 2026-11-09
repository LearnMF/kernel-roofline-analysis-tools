"""第 6 章插图：KDA 在 BW1000 上的四个下限瀑布图（16K/TP8 与 16K/TP2，P-88，2026-10-17）。
数据：老文档 2.1.5.1（理论 / 拆分 / 满载 / 实测）。用法：python3 plot_ch6_bounds.py -> fig6_1_kda_bounds_waterfall.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))
CASES = [("16K/TP8（每卡 12 个 head）", [0.83, 3.35, 4.41, 5.32]),
         ("16K/TP2（每卡 48 个 head）", [2.62, 13.37, 17.99, 22.35])]
NAMES = ["理论下限", "拆分下限", "满载下限", "实测"]
OVH = ["kernel 间交接", "kernel 内冗余", "硬件空等"]
COL = ["#2ca02c", "#d62728", "#ff7f0e", "#1f77b4"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), dpi=150)
for ax, (title, v) in zip(axes, CASES):
    ax.bar(0, v[0], color=COL[0], width=0.6)
    ax.text(0, v[0], f"{v[0]:.2f} ms", ha="center", va="bottom", fontsize=10)
    for i in range(1, 4):
        ax.bar(i, v[i] - v[i - 1], bottom=v[i - 1], color=COL[i], width=0.6)
        ax.bar(i, v[i - 1], color="#dddddd", width=0.6)
        ax.text(i, v[i], f"{v[i]:.2f} ms", ha="center", va="bottom", fontsize=10)
        ax.text(i, (v[i] + v[i - 1]) / 2, f"{OVH[i - 1]}\n×{v[i] / v[i - 1]:.2f}", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_xticklabels(NAMES)
    ax.set_ylabel("时间（ms）")
    ax.set_ylim(0, v[3] * 1.15)
    ax.set_title(f"{title}：实测 / 理论 = {v[3] / v[0]:.1f}")
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("图 6-1　KDA 前向+反向在 BW1000 上的四个下限与三项开销（P-88，2026-10-17）")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig6_1_kda_bounds_waterfall.png"))
print("ok")
