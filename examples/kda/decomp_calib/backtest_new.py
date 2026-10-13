#!/usr/bin/env python3
"""Re-run the decomp_eval backtests with the new calibrated terms wired in via
monkeypatch (decomp_eval.py itself is owned by the main agent).

    python3 backtest_new.py

Prints "experiment | old pred | new pred | measured" three-column tables:
  - P-55 (F3a): s4b loop inflation (48 live VGPR) + icache factor
  - P-49: 64K/48 L2-capacity term on the narrow recurrence
  - every original backtest re-evaluated with lam_scale where res changes
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "kda"))
sys.path.insert(0, str(ROOT))

import decomp_eval as de
import examples.kda.decomp_terms as dt


# ----------------------------------------------------------------- P-55 (F3a)
def bt_p55(shape, meas):
    """P-55 = cand_f3a realized: s4b gains dAqk compute, +~48 live VGPRs in the
    prologue, 456->499 main-loop insts, code 4.8->~8.7 KB.  New terms:
    loop_inflation on s4b.lam + icache_factor on its cycle rate."""
    ks = de.load(shape)
    new_k, _, _ = de.cand_f3a(ks, *map(int, shape.split("_")))
    s4b_new = new_k["wy_s4b#1"]
    # new terms tax EVERY cycle of the changed kernel (fetch stalls + spill
    # rematerialization slow issue, not just latency) -> multiplier on t()
    fac = dt.loop_inflation("wy_s4b", 48) * dt.icache_factor(8700, n_paths=4, ctas_per_cu=2)
    t_s4b_new = s4b_new.t() * fac
    s4b_old = ks["wy_s4b#1"]
    # kernel-level ratio (what P-55 reported): production s4b / changed s4b
    pred = s4b_old.t() / t_s4b_new
    ks2 = de.load(shape)
    nk2, _, _ = de.cand_f3a(ks2, *map(int, shape.split("_")))
    pred_old = ks2["wy_s4b#1"].t() / nk2["wy_s4b#1"].t()
    return dict(name=f"P-55/F3a s4b dAqk recompute ({shape})",
                pred_old=pred_old, pred=pred, meas=meas,
                detail=f"s4b t x{dt.loop_inflation('wy_s4b', 48):.3f} reg x"
                       f"{dt.icache_factor(8700, 4, 2):.3f} icache")


# ----------------------------------------------------------------- P-49 + L2
def bt_p49_l2(shape, meas, shared=(0.5, 0.4)):
    """P-49 with the L2-capacity term: the counterfactual narrow recurrence
    reads l2_extra_hbm_bytes() more than its interface requirement."""
    ks = de.load(shape)
    T, H = map(int, shape.split("_"))
    out_old, out_new = [], []
    for name, s_ in zip(("fwd_h_v8o4#1", "fwd_h_v8o4#2"), shared):
        if name in ks:                                   # prod = narrow
            k = ks[name]
            w = de.geometry(k, 0.5, 1.8, floor_factor=1 - s_ / 2, res_cap=2)
            k2 = de.replace(k, lam=k.lam * dt.l2_lam_factor("fwd_h_v8o4", shape, "narrow"))
        else:                                            # prod = wide
            w = ks[name.replace("v8o4", "v8o4w")]
            k = de.geometry(w, 2.0, 1 / 1.8, floor_factor=1 / (1 - s_ / 2), res_cap=4)
            k2 = de.replace(k, lam=k.lam * dt.l2_lam_factor("fwd_h_v8o4", shape, "narrow"))
        out_old.append(k.t() / w.t())
        out_new.append(k2.t() / w.t())
    return dict(name=f"P-49 64-col recurrences +L2 ({shape}) #1/#2",
                pred_old=out_old, pred=out_new, meas=meas,
                detail="narrow side now carries the L2-thrash extra reads")


def main():
    print("== P-55 (F3a): measured ratios (old/new vs 0.80-0.94) ==")
    for sh, m in (("16384_12", 0.80), ("16384_48", 0.82), ("65536_12", 0.81), ("8192_96", 0.94)):
        r = bt_p55(sh, m)
        print(f"  {r['name']:44s} old={r['pred_old']:.3f} new={r['pred']:.3f} meas={r['meas']:.2f}  {r['detail']}")
    print("== P-49 with L2 term ==")
    for sh, m in (("16384_48", ((1.01, 1.01), (0.91, 0.91))), ("32768_48", ((1.12, 1.12), (1.02, 1.02))),
                  ("65536_48", ((1.40, 1.40), (1.12, 1.12))), ("8192_96", ((0.90, 0.90), (0.68, 0.68)))):
        r = bt_p49_l2(sh, m)
        po = " / ".join(f"{x:.3f}" for x in r["pred_old"])
        pn = " / ".join(f"{x:.3f}" for x in r["pred"])
        ms = " / ".join(f"{x[0]:.2f}" for x in m)
        print(f"  {r['name']:44s} old={po:12s} new={pn:12s} meas={ms}")
    print("== original backtests (unchanged model, sanity) ==")
    de.backtest()


if __name__ == "__main__":
    main()
