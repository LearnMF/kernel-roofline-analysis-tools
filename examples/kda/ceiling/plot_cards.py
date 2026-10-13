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
                 f"(total {tot:.1f} ms; G2 fp 35b7bd3a8862; {M['id']}; 2026-10-13)")
    ax.grid(True, which="both", ls=":", lw=0.5)
    ax.legend(loc="lower right", fontsize=8)

    # ---- right panel: measured utilization per kernel (task 7.3 right column)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from kda_bound import bounds as algo_bounds
    ks = sorted(c["cards"].items(), key=lambda x: -x[1]["t_us"])
    names = [k.replace("#1", "") for k, _ in ks]
    ypos = np.arange(len(ks))
    hbw = M["peaks"]["hbm_read"]["value"] * 1e9
    l2w = M["peaks"]["l2_read"]["value"] * 1e9
    ldsw = M["peaks"]["lds_read"]["value"] * 1e9
    rows = []
    for k, v in ks:
        t = v["t_us"] * 1e-6
        chan = lambda b: (b or 0) / t / 1e9                    # bytes -> GB/s
        u_hb = (v["hbm_rd"] + v["hbm_wr"]) / t / hbw
        u_l2 = v["tcc_req"] * 64 / t / l2w
        u_ld = v["lds_idx"] * 16 / t / ldsw
        u_cn = v["conf"] / (v["act"] * 1.5e9 * t) if v.get("conf") else 0.0
        u_ch = (v["chain_active"] / (v["act"] * 1.5e9 * t)
                if v.get("chain_active") and k.split("#")[0].startswith(("fwd_h", "dhu"))
                else 0.0)
        rows.append(dict(name=k.replace("#1", ""), u_hb=u_hb, u_l2=u_l2, u_ld=u_ld,
                         u_ta=v["u_ta_act"], u_vl=v["u_valu_act"], u_mm=v["u_mmop_act"],
                         u_cn=u_cn, u_ch=u_ch, primary=v["primary"]))
    bars = [("u_hb", "HBM", "tab:red"), ("u_l2", "L2", "tab:cyan"),
            ("u_ld", "LDS", "tab:purple"), ("u_ta", "TA", "tab:blue"),
            ("u_vl", "VALU", "tab:orange"), ("u_mm", "MMAC", "tab:green"),
            ("u_cn", "LDS冲突", "0.6"), ("u_ch", "chain", "tab:brown")]
    w = 0.10
    pri_map = {"mem": "u_hb", "lds": "u_ld", "ta": "u_ta", "valu": "u_vl",
               "mmac": "u_mm", "mmop": "u_mm", "chain": "u_ch", "l2": "u_l2"}
    for i, (f, lab, colr) in enumerate(bars):
        vals = [r[f] for r in rows]
        for j, (x0, vv) in enumerate(zip(ypos + (i - 3.5) * w, vals)):
            if vv < 0.005:
                continue
            pri = pri_map.get(rows[j]["primary"]) == f
            ax2.barh(x0, min(vv, 1.15), height=w, color=colr, alpha=0.85,
                     edgecolor="k" if pri else "none", lw=2.2 if pri else 0,
                     label=lab if j == 0 else None)
            if pri:
                ax2.text(min(vv, 1.12) + 0.01, x0, rows[j]["primary"],
                         va="center", fontsize=6, fontweight="bold", color="0.3")
    for ref in (0.85, 0.6):
        ax2.axvline(ref, color="k", ls=":" if ref == 0.6 else "--", lw=0.9)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([f"{r['name']}  {v['t_us'] / 1e3:.1f}ms"
                         for r, (k, v) in zip(rows, ks)], fontsize=7.5)
    ax2.set_xlim(0, 1.3)
    ax2.set_xlabel("measured utilization (channels: vs attainable; units: active-CU normalized)")
    ax2.set_title("Fig A-R — per-kernel utilization")
    ax2.legend(fontsize=6.5, loc="lower right", ncol=2)
    ax2.grid(axis="x", ls=":", lw=0.5)

    # ---- left panel: operator-level square point
    b = algo_bounds("BW1000", c["T"], c["H"])
    hbm_tot = sum(v["hbm_rd"] + v["hbm_wr"] for v in c["cards"].values())
    t_tot = sum(v["t_us"] for v in c["cards"].values())
    ax.plot([b["flops"] / hbm_tot], [b["flops"] / (t_tot * 1e-6) / 1e12], "s",
            color="tab:blue", ms=12, mfc="none", mew=2.5, zorder=10)
    ax.annotate("operator", (b["flops"] / hbm_tot, b["flops"] / (t_tot * 1e-6) / 1e12),
                textcoords="offset points", xytext=(8, -4), fontsize=9,
                fontweight="bold", color="tab:blue")

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
        ta = [v.get("ta_amp") or 0 for k, v in rows if v.get("w_gap")]
        x = np.arange(len(names))
        ax.bar(x - 0.2, ta, 0.4, color="tab:blue",
               label="TA amplification (measured TA-busy / ideal-TA)")
        ax.bar(x + 0.2, wb, 0.4, color="tab:red", label="w_gap (HBM bytes / interface)")
        ax.axhline(1.0, color="0.4", ls="--", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7.5)
        ax.set_title(f"{c['T'] // 1024}K/H{c['H']} work amplification", fontsize=9, loc="left")
        ax.set_yscale("log"); ax.set_ylim(0.3, 20)
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
    ss = sorted(shapes, key=lambda s: (cards[s]["T"], cards[s]["H"]))
    kernels = sorted({k for s in ss for k in cards[s]["cards"]},
                     key=lambda k: -max(cards[s]["cards"].get(k, {}).get("t_us", 0)
                                        for s in ss))
    im = np.full((len(kernels), len(ss)), np.nan)
    for j, s in enumerate(ss):
        tot = sum(v["t_us"] for v in cards[s]["cards"].values())
        for i, k in enumerate(kernels):
            v = cards[s]["cards"].get(k)
            im[i, j] = v["t_us"] / tot if v else np.nan
    bind_cols = {"mem": "tab:red", "ta": "tab:blue", "valu": "tab:orange",
                 "lds": "tab:purple", "chain": "tab:brown", "mmop": "tab:green",
                 "mmac": "tab:green", "issue": "tab:blue"}
    fig, ax = plt.subplots(figsize=(13, 0.55 * len(kernels) + 2.5))
    msk = np.ma.masked_invalid(im)
    ax.imshow(msk, aspect="auto", cmap="viridis", vmin=0, vmax=im[~np.isnan(im)].max())
    from matplotlib.patches import Rectangle
    for i in range(len(kernels)):
        for j, s in enumerate(ss):
            if np.isnan(im[i, j]):
                continue
            v = cards[s]["cards"][kernels[i]]
            pct = im[i, j] * 100
            lum = 0.299 * 0.267 + 0.587 * 0.005 + 0.114 * 0.329   # viridis-ish luminance
            frac = im[i, j] / im[~np.isnan(im)].max()
            tc = "k" if frac > 0.55 else "w"
            ax.text(j, i, f"{pct:.0f}%\n{v['primary'][:3].upper()}.{v['u_primary']:.2f}",
                    ha="center", va="center", fontsize=6.8, color=tc)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor=bind_cols.get(v["binding"], "0.8"), lw=1.8))
    ax.set_xticks(range(len(ss)))
    ax.set_xticklabels([f"{cards[s]['T'] // 1024}K/H{cards[s]['H']}" for s in ss],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(kernels)))
    ax.set_yticklabels([k.replace("#1", "") for k in kernels], fontsize=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fill=False, edgecolor=cc, label=b)
                       for b, cc in bind_cols.items() if b != "mmac"],
              fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("Figure D — kernel time share per shape (cell: % share + primary.util; "
                 "border color = binding resource)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_gap_matrix(cards, out):
    """rows=kernels, cols=shapes; cell = gap types + potential gain us vs iface floor.
    Items excluded by ablation evidence (R42/R43/P54b) are hatched gray."""
    ss = sorted(cards, key=lambda s: (cards[s]["T"], cards[s]["H"]))
    kernels = sorted({k for s in ss for k in cards[s]["cards"]},
                     key=lambda k: -max(cards[s]["cards"].get(k, {}).get("t_us", 0)
                                        for s in ss))
    excluded_full = {"dav", "wu", "fwd_prep_a", "fwd_prep_b"}          # R42/43 ablated: no gain
    excluded_part = {"fwd_h_v8o4", "fwd_h_v8o4w", "dhu_v8b", "dhu_v8b4"}  # conflict part excluded
    fig, ax = plt.subplots(figsize=(13, 0.5 * len(kernels) + 2))
    gain_max = 0.0
    cells = {}
    for i, k in enumerate(kernels):
        for j, s in enumerate(ss):
            v = cards[s]["cards"].get(k)
            if not v:
                continue
            gain = max(v["t_us"] - (v.get("f_iface_us") or v["t_us"]), 0)
            cells[(i, j)] = (v.get("gap_types", "—"), gain, k)
            gain_max = max(gain_max, gain)
    ax.imshow(np.zeros((len(kernels), len(ss))), cmap="Blues", vmin=0,
              vmax=1, aspect="auto", alpha=0.0)
    from matplotlib.patches import Rectangle
    for (i, j), (gt, gain, k) in cells.items():
        a = min(gain / gain_max, 1.0)
        base = k.split("#")[0]
        fc = (0.2, 0.4, 0.8, 0.15 + 0.55 * a)
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=fc, edgecolor="0.7", lw=0.4))
        if base in excluded_full:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="none",
                                   edgecolor="0.4", hatch="///", lw=0))
        elif base in excluded_part:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 0.45, facecolor="none",
                                   edgecolor="0.4", hatch="///", lw=0))
        ax.text(j, i - 0.18, gt, ha="center", va="center", fontsize=6.5, fontweight="bold")
        ax.text(j, i + 0.22, f"{gain / 1e3:.2f}ms", ha="center", va="center", fontsize=5.8,
                color="0.25")
    ax.set_xticks(range(len(ss)))
    ax.set_xticklabels([f"{cards[s]['T'] // 1024}K/H{cards[s]['H']}" for s in ss],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(kernels)))
    ax.set_yticklabels([k.replace("#1", "") for k in kernels], fontsize=8)
    ax.set_xlim(-0.5, len(ss) - 0.5); ax.set_ylim(len(kernels) - 0.5, -0.5)
    ax.set_title("fig_gap_matrix — gap types + potential gain (ms vs interface floor); "
                 "hatched = ablation-excluded items (R42/43)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def figd_main():
    cards = json.load(open(RES / "cards.json"))
    figd(list(cards), cards, RES / "figD.png")
    fig_gap_matrix(cards, RES / "fig_gap_matrix.png")
    print("figD + gap_matrix done")


figd_main()


def fige(cards, out):
    """fig_op_gap: per shape 4 bars — measured (stacked by kernel), Σstream floor,
    Σinterface floor, operator bound. Gap shares annotated on top."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from kda_bound import bounds as algo_bounds
    ss = sorted(cards, key=lambda s: (cards[s]["T"], cards[s]["H"]))
    meas = np.array([sum(v["t_us"] for v in cards[s]["cards"].values()) / 1e3 for s in ss])
    flor = np.array([sum(v["floor_us"] for v in cards[s]["cards"].values()) / 1e3 for s in ss])
    iface = np.array([sum(v.get("f_iface_us") or 0 for v in cards[s]["cards"].values()) / 1e3
                      for s in ss])
    bnd = np.array([algo_bounds("BW1000", cards[s]["T"], cards[s]["H"])["bound"] * 1e3
                    for s in ss])
    x = np.arange(len(ss))
    fig, ax = plt.subplots(figsize=(15, 6))
    # measured: stacked per-kernel segments (top-6 named, rest gray)
    top_names = ["wy_s4b", "wy_s4a", "bwd_tail", "wu", "dhu", "fwd_h", "prep"]
    palette = ["tab:red", "tab:orange", "gold", "tab:purple", "tab:brown",
               "tab:pink", "tab:olive"]
    for i, s in enumerate(ss):
        bot = 0.0
        groups = {n: 0.0 for n in top_names}
        for k, v in cards[s]["cards"].items():
            base = next((n for n in top_names if k.startswith(n)), "other")
            groups[base] = groups.get(base, 0) + v["t_us"] / 1e3
        for gi, (n, tv) in enumerate(groups.items()):
            col = palette[top_names.index(n)] if n in top_names else "0.75"
            ax.bar(x[i] - 0.3, tv, 0.26, bottom=bot, color=col, alpha=0.9,
                   label=n if i == 0 else None)
            bot += tv
    ax.bar(x, flor, 0.20, color="tab:blue", alpha=0.9, label="sum stream floor")
    ax.bar(x + 0.24, iface, 0.20, color="tab:cyan", alpha=0.9, label="sum interface floor")
    ax.bar(x + 0.46, bnd, 0.20, color="tab:green", alpha=0.9, label="first-principles bound")
    for i in range(len(ss)):
        r_stall = (meas[i] - flor[i]) / meas[i]
        r_strm = (flor[i] - iface[i]) / meas[i]
        r_ifc = (iface[i] - bnd[i]) / meas[i]
        ax.text(x[i] - 0.3, meas[i] * 1.05,
                f"stall {r_stall * 100:.0f}% | strm x{flor[i] / iface[i]:.2f} | ifc x{iface[i] / bnd[i]:.1f}",
                ha="center", fontsize=6.3, rotation=0)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{cards[s]['T'] // 1024}K/H{cards[s]['H']}" for s in ss],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ms (log scale)")
    ax.set_title("fig_op_gap — measured (stacked) / sum stream floor / sum interface floor / operator bound")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(axis="y", ls=":", lw=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


fige(cards=json.load(open(RES / "cards.json")), out=RES / "fig_op_gap.png")
print("fig_op_gap done")
