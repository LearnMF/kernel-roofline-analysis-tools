"""Figure A (per shape): hierarchical roofline with each kernel's LDS/L2/HBM points
at its exact ISA-typed MMAC FLOPs.  Same conventions as hier_roofline.py:
roof lines (measured attainable), ridge dots, three point types, non-overlapping
labels, gray hatch + dashed "—" handling for unreliable (t<10us) or missing data.

    python3 plot_cards.py            # every shape in cards.json -> figA_<T>_<H>.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                          # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "results" / "kda-ceiling-cards-20261013"
M = json.load(open(ROOT / "machines" / "gfx936-bw1000.json"))
PK = M["peaks"]

MEM = [("LDS", PK["lds_read"]["value"], "tab:purple"),
       ("L2", PK["l2_read"]["value"], "tab:blue"),
       ("HBM", PK["hbm_read"]["value"], "tab:red")]
COMP = [("bf16 MMAC", PK["mmac_bf16"]["value"], "k"),
        ("tf32 MMAC", PK["mmac_tf32"]["value"], "0.35"),
        ("fp32 MMAC (16x16x8)", PK["mmac_f32x8"]["value"], "tab:green"),
        ("fp32 VALU FMA", PK["valu_fma_f32"]["value"], "tab:orange")]
LVL = [("lds", "d"), ("l2", "v"), ("hbm", "o")]


def figa(shape: str, c: dict, out: Path):
    fig = plt.figure(figsize=(15.5, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    xs = np.logspace(-2, 5, 400)
    top = max(t for _, t, _ in COMP)
    for name, g, col in MEM:
        ax.plot(xs, np.minimum(g * xs / 1e3, top), "-", color=col, lw=2,
                label=f"{name} {g / 1e3:.2f} TB/s")
    for name, tf, col in COMP:
        ax.axhline(tf, color=col, ls="--", lw=1.2)
        ax.text(1.1e4, tf * 1.05, f"{name} {tf:.0f} TF", ha="right", fontsize=8, color=col)
    for name, g, col in MEM:                                    # ridge points
        for cname, tf, ccol in (COMP[0], COMP[2]):
            r = tf * 1e3 / g
            ax.plot([r], [tf], "o", color=col, ms=5)
            ax.annotate(f"{r:.0f}", (r, tf), textcoords="offset points",
                        xytext=(4, -12), fontsize=8, color=col)
    # ---- kernel points (left panel)
    movers, pts, legend_rows = [], [], []
    for key, v in sorted(c["cards"].items(), key=lambda x: -x[1]["t_us"]):
        t = v["t_us"]
        unreliable = t < 10
        fl = v["flops"]
        if fl < 1e8:
            movers.append((key, t, unreliable))
            continue
        y = fl / (t * 1e-6) / 1e12
        xs3 = [(fl / b, mk) for lv, mk in LVL if (b := v.get(lv) or 0) > 0]
        if xs3:
            pts.append((key, xs3, y, t, unreliable))
    pts.sort(key=lambda p: p[2])
    for key, xs3, y, t, unreliable in pts:
        xlist = [x for x, _ in xs3]
        col = "0.55" if unreliable else "tab:red"
        if len(xlist) > 1:
            ax.plot(sorted(xlist), [y] * len(xlist), "-", color=col, lw=0.7, alpha=0.4)
        for x, mk in xs3:
            big = (mk == "o")                       # HBM point: largest, filled
            ax.plot([x], [y], mk, color=col, ms=7.5 if big else 5,
                    mfc="none" if unreliable else col, mew=1.8 if big else 1.2, alpha=0.9)
        x0 = next((x for x, m2 in xs3 if m2 == "o"), xlist[0])
        tag = f"K{len(legend_rows) + 1}"
        ax.annotate(tag, (x0, y), textcoords="offset points", xytext=(5, 4),
                    fontsize=7.5, fontweight="bold", color=col,
                    alpha=0.6 if unreliable else 1.0)
        legend_rows.append(f"{tag} {key.replace('#1', ''):18s} {t:7.0f}us"
                           + (" *" if unreliable else ""))
    for k, t, unreliable in movers:
        tag = f"K{len(legend_rows) + 1}"
        legend_rows.append(f"{tag} {k.replace('#1', ''):18s} {t:7.0f}us  (no MMAC)")
    ax.text(1.02, 0.97, "\n".join(legend_rows), transform=ax.transAxes,
            fontsize=7, family="monospace", va="top", color="0.25")
    if any(u for _, _, u in movers):
        ax.text(0.02, 0.025, "* <10us: counters unreliable", transform=ax.transAxes,
                fontsize=7.5, color="0.4")
    ax.plot([], [], "o", color="0.4", ms=7, label="kernel @ HBM AI")
    ax.plot([], [], "v", color="0.4", ms=5, label="kernel @ L2 AI")
    ax.plot([], [], "d", color="0.4", ms=5, label="kernel @ LDS AI")
    T, H = c["T"], c["H"]
    tot = sum(v["t_us"] for v in c["cards"].values()) / 1e3
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.5, 2e4); ax.set_ylim(0.3, 900)
    ax.set_xlabel("Arithmetic intensity (FLOP / byte of that level)")
    ax.set_ylabel("Achieved performance (TFLOPS)")
    ax.set_title(f"Fig A-L — {T // 1024}K/H{H} hierarchical roofline "
                 f"(kernel total {tot:.1f} ms; P-52 {M['id']})")
    ax.grid(True, which="both", ls=":", lw=0.5)
    ax.legend(loc="lower right", fontsize=8)

    # ---- right panel: measured utilization per kernel (active-CU normalized)
    ks = sorted(c["cards"].items(), key=lambda x: x[1]["t_us"])
    names = [k.replace("#1", "") for k, _ in ks]
    ypos = np.arange(len(ks))
    u_ta = [v["u_ta_act"] for _, v in ks]
    u_vl = [v["u_valu_act"] for _, v in ks]
    u_ld = [v["u_lds_act"] for _, v in ks]
    u_mm = [v["u_mmop_act"] for _, v in ks]
    u_hb = [(v["hbm_rd"] + v["hbm_wr"]) / max(1e-9, v["t_us"] * 1e-6) / 1.34e12
            for _, v in ks]
    w = 0.16
    for i, (uu, lab, colr) in enumerate([(u_hb, "HBM", "tab:red"),
                                       (u_ld, "LDS", "tab:purple"),
                                       (u_ta, "TA issue", "tab:blue"),
                                       (u_vl, "VALU", "tab:orange"),
                                       (u_mm, "MMAC", "tab:green")]):
        ax2.barh(ypos + (i - 2) * w, uu, height=w, color=colr, alpha=0.85, label=lab)
    ax2.axvline(1.0, color="k", ls="--", lw=1)
    ax2.set_yticks(ypos); ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlim(0, 1.15)
    ax2.set_xlabel("utilization (vs measured attainable, active-CU normalized)")
    ax2.set_title("Fig A-R — measured utilization")
    ax2.legend(fontsize=7, loc="lower right")
    ax2.grid(axis="x", ls=":", lw=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    cards = json.load(open(RES / "cards.json"))
    for s, c in cards.items():
        figa(s, c, RES / f"figA_{s}.png")
        print("figA", s)


if __name__ == "__main__":
    main()


BIND_COL = {"mem": "tab:red", "compute": "k", "issue": "tab:olive", "chain": "tab:purple"}


def figb(shapes, cards, out):
    """Per shape: per-kernel measured time split floor vs above-floor overhead."""
    n = len(shapes)
    fig, axs = plt.subplots(n, 1, figsize=(11.5, 2.1 * n + 1), sharex=False)
    if n == 1:
        axs = [axs]
    for ax, s in zip(axs, shapes):
        c = cards[s]
        rows = sorted(c["cards"].items(), key=lambda x: x[1]["t_us"])
        for i, (k, v) in enumerate(rows):
            f = v["floor_us"]
            ax.barh(i, f, color=BIND_COL[v["binding"]], alpha=0.85)
            ax.barh(i, v["t_us"] - f, left=f, color="0.75", alpha=0.8)
            ax.text(v["t_us"], i, f" {v['t_us']:.0f}", va="center", fontsize=7)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([k for k, _ in rows], fontsize=8)
        tot = sum(v["t_us"] for v in c["cards"].values()) / 1e3
        fl = sum(v["floor_us"] for v in c["cards"].values()) / 1e3
        ax.set_title(f"{c['T'] // 1024}K/H{c['H']}: {tot:.1f} ms (sum-floor {fl:.1f} ms)",
                     fontsize=9, loc="left")
        ax.grid(True, axis="x", ls=":", lw=0.5)
    from matplotlib.patches import Patch
    hs = [Patch(color=col, label=f"floor: {b}") for b, col in BIND_COL.items()]
    hs.append(Patch(color="0.75", label="above floor"))
    axs[0].legend(handles=hs, fontsize=7, ncol=5, loc="upper center",
                  bbox_to_anchor=(0.5, 1.5 + 0.12 * (5 - n)))

    fig.suptitle("Figure B — measured vs per-kernel floor (gap = removable above-floor time)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def figc(shapes, cards, out):
    """Per shape: actual/required ratios (HBM bytes; VMEM instructions)."""
    n = len(shapes)
    fig, axs = plt.subplots(n, 1, figsize=(11.5, 1.9 * n + 1))
    if n == 1:
        axs = [axs]
    for ax, s in zip(axs, shapes):
        c = cards[s]
        rows = sorted(c["cards"].items(), key=lambda x: -x[1]["t_us"])
        names = [k for k, v in rows if v.get("w_gap")]
        wb = [v["w_gap"] for k, v in rows if v.get("w_gap")]
        vi = [v["vmem"] / (v["req_bytes"] / 1024) for k, v in rows if v.get("w_gap")]
        x = np.arange(len(names))
        ax.bar(x - 0.2, wb, 0.4, color="tab:red", label="HBM bytes / interface")
        ax.bar(x + 0.2, vi, 0.4, color="tab:olive", label="VMEM insts / 1KB-pattern bound")
        ax.axhline(1.0, color="0.4", ls="--", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7.5)
        ax.set_title(f"{c['T'] // 1024}K/H{c['H']} work amplification", fontsize=9, loc="left")
        ax.set_yscale("log"); ax.set_ylim(0.3, 10)
        ax.grid(True, axis="y", ls=":", lw=0.5)
    axs[-1].legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.5))
    fig.suptitle("Figure C — actual / required work multipliers")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def figb_figc():
    cards = json.load(open(RES / "cards.json"))
    shapes = sorted(cards, key=lambda s: (cards[s]["T"], cards[s]["H"]))
    figb(shapes, cards, RES / "figB.png")
    figc(shapes, cards, RES / "figC.png")
    print("figB/figC:", ", ".join(shapes))


figb_figc()


def figd(shapes, cards, out):
    """16 shapes x (kernel, binding) heatmap: share of kernel time by binding."""
    bind_cols = {"mem": "#d62728", "compute": "#222222", "issue": "#bcbd22", "chain": "#9467bd"}
    kernels = ["wy_s4b#1", "wy_s4a#1", "bwd_tail#1", "wu#1", "dhu_v8b4#1", "dhu_v8b#1",
               "fwd_prep_a#1", "fwd_prep_b#1", "dav#1", "gate#1", "l2n_apply#1",
               "fwd_h_v8o4#1", "fwd_h_v8o4w#1", "fwd_h_v8o4#2", "fwd_h_v8o4w#2",
               "exp2#1", "bwd_tail_reduce#1"]
    ss = sorted(shapes, key=lambda s: (cards[s]["T"], cards[s]["H"]))
    fig, ax = plt.subplots(figsize=(11, 6.5))
    im = np.zeros((len(kernels), len(ss)))
    for j, s in enumerate(ss):
        c = cards[s]
        tot = sum(v["t_us"] for v in c["cards"].values())
        for i, k in enumerate(kernels):
            v = c["cards"].get(k)
            im[i, j] = v["t_us"] / tot if v else np.nan
    ax.imshow(np.ma.masked_invalid(im), aspect="auto", cmap="viridis", vmin=0, vmax=im[~np.isnan(im)].max())
    for i in range(len(kernels)):
        for j, s in enumerate(ss):
            if not np.isnan(im[i, j]):
                v = cards[s]["cards"][kernels[i]]
                ax.text(j, i, f"{im[i, j] * 100:.0f}", ha="center", va="center",
                        fontsize=7, color=bind_cols.get(v["binding"], "w"))
    ax.set_xticks(range(len(ss)))
    ax.set_xticklabels([f"{cards[s]['T'] // 1024}K/H{cards[s]['H']}" for s in ss],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(kernels)))
    ax.set_yticklabels([k.replace("#1", "") for k in kernels], fontsize=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=cc, label=b) for b, cc in bind_cols.items()],
              fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("Figure D — kernel time share (%) per shape; text color = binding resource")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def figd_main():
    cards = json.load(open(RES / "cards.json"))
    figd(list(cards), cards, RES / "figD.png")
    print("figD done")


figd_main()


def fige(cards, out):
    """Operator-level gap: measured vs sum-of-floors vs first-principles bound."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from kda_bound import bounds as algo_bounds
    ss = sorted(cards, key=lambda s: (cards[s]["T"], cards[s]["H"]))
    meas = np.array([sum(v["t_us"] for v in cards[s]["cards"].values()) / 1e3 for s in ss])
    flor = np.array([sum(v["floor_us"] for v in cards[s]["cards"].values()) / 1e3 for s in ss])
    bnd = np.array([algo_bounds("BW1000", cards[s]["T"], cards[s]["H"])["bound"] * 1e3
                    for s in ss])
    x = np.arange(len(ss))
    fig, ax = plt.subplots(figsize=(13.5, 5.5))
    ax.bar(x - 0.28, meas, 0.26, color="tab:red", alpha=0.85, label="measured Σ kernels")
    ax.bar(x, flor, 0.26, color="tab:blue", alpha=0.85,
           label="Σ existing-instruction-stream floors")
    ax.bar(x + 0.28, bnd, 0.26, color="tab:green", alpha=0.85,
           label="operator first-principles bound")
    for i in range(len(ss)):
        ax.text(i, flor[i] * 1.03, f"{meas[i] / flor[i]:.2f}x", ha="center", fontsize=7)
        ax.text(i + 0.28, bnd[i] * 1.03, f"{flor[i] / bnd[i]:.1f}x", ha="center",
                fontsize=7, color="tab:green")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{cards[s]['T'] // 1024}K/H{cards[s]['H']}" for s in ss],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ms (log scale)")
    ax.set_title("Figure E — operator-level gap: measured / Σ instruction-stream floors / operator bound")
    ax.legend(fontsize=8)
    ax.grid(axis="y", ls=":", lw=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


fige(cards=json.load(open(RES / "cards.json")), out=RES / "figE.png")
print("figE done")
