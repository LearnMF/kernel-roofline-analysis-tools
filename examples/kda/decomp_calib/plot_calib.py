#!/usr/bin/env python3
"""Calibration figures for the decomp probes (task deliverable §3.4).
    python3 plot_calib.py            # writes results/kda-decomp-calib-20261013/fig/
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "kda-decomp-calib-20261013"
RAW = RES / "raw"
CAL = json.load(open(RES / "calib.json"))


def fig_p1():
    """t(residency) and lambda(residency) curves per kernel, both shapes."""
    rows = CAL["p1"]["rows"]
    shapes = sorted({r["shape"] for r in rows})
    kernels = sorted({r["kernel"] for r in rows})
    fig, axs = plt.subplots(2, len(shapes), figsize=(6 * len(shapes), 7),
                            squeeze=False, sharey="row")
    for j, s in enumerate(shapes):
        for k in kernels:
            pts = sorted([(r["res1"], r["t_us"], r["lam_us"]) for r in rows
                          if r["shape"] == s and r["kernel"] == k])
            if not pts:
                continue
            res = [p[0] for p in pts]
            axs[0][j].plot(res, [p[1] / 1e3 for p in pts], "o-", label=k)
            axs[1][j].plot(res, [p[2] for p in pts], "s-", label=k)
        for a, yl in ((axs[0][j], "kernel time (ms)"), (axs[1][j], "lambda = t / rounds (us)")):
            a.set_xlabel("resident CTAs / CU")
            a.set_ylabel(yl)
            a.set_title(f"{int(s.split('_')[0]) // 1024}K/H{s.split('_')[1]}")
            a.grid(ls=":", lw=0.5)
            a.invert_xaxis()
        axs[0][j].legend(fontsize=7)
    fig.suptitle("P1: t(residency) and lambda(residency) — LDS topup probe")
    fig.tight_layout()
    fig.savefig(RES / "fig" / "p1_residency.png", dpi=150)
    plt.close(fig)


def fig_p2():
    """VGPR / loop insts / time vs injected pinned registers."""
    p2 = CAL["p2"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    for kern, d in p2["per_kernel"].items():
        ns = sorted(int(n) for n in d["vgpr"])
        axs[0].plot(ns, [d["vgpr"][str(n)] for n in ns], "o-", label=kern)
        axs[1].plot(ns, [d["loop_insts"][str(n)] for n in ns], "o-", label=kern)
        for sh, tt in d["t"].items():
            axs[2].plot(ns, [tt[str(n)] / 1e3 for n in ns], "o-",
                        label=f"{kern} {sh}")
    axs[0].set_ylabel("VGPR count"); axs[0].axhline(256, color="r", ls="--", lw=1)
    axs[1].set_ylabel("main-loop instructions")
    axs[2].set_ylabel("kernel time (ms)")
    for a in axs:
        a.set_xlabel("pinned live registers (N)")
        a.grid(ls=":", lw=0.5)
        a.legend(fontsize=7)
    fig.suptitle("P2: register pressure — resources / loop / time")
    fig.tight_layout()
    fig.savefig(RES / "fig" / "p2_regpress.png", dpi=150)
    plt.close(fig)


def fig_p3():
    """cycles/instruction vs total code footprint; mark s4b prod / P-55."""
    p3 = CAL["p3"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for g, col in (("80", "tab:blue"), ("160", "tab:orange"), ("240", "tab:green")):
        pts = sorted(p3["raw"][g].items(), key=lambda x: int(x[0]))
        ax.plot([int(m) * 16 for m, _ in pts],
                [v for _, v in pts], "o-", label=f"{int(g) // 80} CTA/CU")
    ax.set_xscale("log")
    ax.set_xlabel("total kernel code footprint (bytes)")
    ax.set_ylabel("cycles per instruction")
    for name, b, col in (("s4b prod", p3["s4b_prod_bytes"], "tab:red"),
                         ("s4b P-55 est", p3["s4b_p55_bytes"], "tab:purple")):
        ax.axvline(b, color=col, ls="--", lw=1.2)
        ax.text(b * 1.1, ax.get_ylim()[0] + 0.5, name, color=col, rotation=90,
                fontsize=8, va="bottom")
    ax.legend()
    ax.grid(ls=":", lw=0.5)
    ax.set_title("P3: icache — cycles/inst vs code size (4 wave paths)")
    fig.tight_layout()
    fig.savefig(RES / "fig" / "p3_icache.png", dpi=150)
    plt.close(fig)


def fig_p4():
    """fwd recurrence L2 hit rate + extra-HBM-read ratio vs shape; model overlay."""
    p4 = CAL["p4"]
    cards = json.load(open(ROOT / "results" / "kda-ceiling-cards-20261013" / "cards.json"))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for kpat, col in (("fwd_h_v8o4#1", "o-"), ("fwd_h_v8o4w#1", "s-")):
        for H, m in ((12, "^"), (24, "v"), (48, "o"), (96, "s")):
            xs, hit, ex = [], [], []
            for T in (8192, 16384, 32768, 65536):
                v = cards[f"{T}_{H}"]["cards"].get(kpat.replace("v8o4", "v8o4"))
                if not v:
                    continue
                xs.append(T / 1024)
                hit.append(1 - v["hbm_rd"] / (v["tcc_req"] * 64))
                ex.append(v["hbm_rd"] / max(v["req_rd"], 1) - 1)
            if xs:
                axs[0].plot(xs, hit, m + "-", label=f"{kpat} H{H}", ms=5)
                axs[1].plot(xs, ex, m + "-", label=f"{kpat} H{H}", ms=5)
    axs[0].set_ylabel("L2 hit rate (1 - HBM_rd / L2_req)")
    axs[1].set_ylabel("extra HBM reads (hbm_rd / req_rd - 1)")
    for a in axs:
        a.set_xlabel("T (K)")
        a.legend(fontsize=6, ncol=2)
        a.grid(ls=":", lw=0.5)
    axs[0].axhline(0, color="k", lw=0.8)
    axs[1].set_title("L2 thrash shows up as HBM re-reads (narrow, H>=48)")
    axs[0].set_title("fwd recurrence pass 1")
    fig.suptitle("P4: L2 capacity — recurrence working set vs T,H")
    fig.tight_layout()
    fig.savefig(RES / "fig" / "p4_l2.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    (RES / "fig").mkdir(parents=True, exist_ok=True)
    fig_p1(); fig_p2(); fig_p3(); fig_p4()
    print("figs written to", RES / "fig")
