"""`python -m kra validate --machine M --profile P.yaml`: shape-only bounds vs measurements.

Takes an hcu-train-simulator operator profile (measured mean_ms per shape) and checks,
for every entry whose timed workload is unambiguous:

  1. bound holds    T_theory <= T_meas            (a violation is a model bug)
  2. efficiency     eta = T_theory / T_meas        (how close the measured code gets)
  3. predictor      floor + T_theory / eta with two machine-wide constants fitted
                    leave-one-out, against the simulator's own tiered estimate
                    6mnk / (peak * eff * tier(min dim))

(3) is the like-for-like comparison: both turn an ideal time into a runtime with fitted
constants; the tiered model uses a 4-step table on min(m,n,k) (whose steps were read off
this same profile), ours uses the bound's structure and two constants.

Bound assumptions for the gemm harness (`y = x @ W; y.float().sum().backward()`): three
GEMM kernels plus the harness's cast to fp32 (2+4 B/elem), sum (4 B/elem) and the cast
of the broadcast gradient back to bf16 (2 B/elem written), each its own launch.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics

from ..ops import Work, get
from .predict import TheoryMachine, _terms

TIER_PEAK_TFLOPS = 480.0  # hcu-train-simulator templates/config.yaml: fp16_tflops
TIER_GEMM_EFF = 0.6       # ... gemm_efficiency


def tier(m: int, n: int, k: int) -> float:
    s = min(m, n, k)
    return 0.15 if s <= 512 else 0.40 if s <= 1024 else 0.75 if s <= 2048 else 1.0


def _key(k: str) -> dict:
    return dict(p.split("=", 1) for p in k.split(","))


def _bound(mach: TheoryMachine, w: Work, dtype: str) -> tuple:
    peak_key, peak = mach.compute_peak(dtype)
    total, by = 0.0, {}
    for _, pw in (w.parts or [("", w)]):
        t = _terms(mach, pw, peak_key, peak)
        b = max(t, key=t.get)
        total += t[b]
        by[b.split(":")[0]] = by.get(b.split(":")[0], 0.0) + t[b]
    return total, max(by, key=by.get)


def _harness(m: int, n: int) -> list:
    mn = float(m * n)
    return [("harness y.float()", Work(bytes_min=6 * mn)),
            ("harness .sum()", Work(bytes_min=4 * mn)),
            ("harness grad cast", Work(bytes_min=2 * mn))]


def gemm_rows(mach: TheoryMachine, profile: dict, harness: bool = True) -> list:
    rows = []
    for key, v in profile.get("operators", {}).get("gemm", {}).items():
        kv = _key(key)
        m, n, k = (int(float(kv[x])) for x in ("m", "n", "k"))
        row = {"key": key, "part": kv.get("part", ""), "M": m, "N": n, "K": k,
               "dtype": kv.get("dtype", "bf16"), "T_meas_us": float(v["mean_ms"]) * 1e3}
        tf = v.get("tflops")
        if tf is None:
            row["excluded"] = "no tflops recorded: not a plain matmul benchmark"
        else:
            ratio = tf * 1e12 * v["mean_ms"] * 1e-3 / (6.0 * m * n * k)
            if abs(ratio - 1.0) > 0.01:
                row["excluded"] = ("timed workload is %.2fx the key's 6mnk flops (grouped/"
                                   "fused or a different shape was timed)" % ratio)
        if kv.get("grouped_gemm") == "True":
            row["excluded"] = "grouped GEMM: per-expert shape is not the key's m,n,k"
        if row.get("excluded"):
            rows.append(row)
            continue
        w = get("gemm").work({"M": m, "N": n, "K": k}, row["dtype"], "fwd+bwd")
        if harness:
            w = Work.sequence(w.parts + _harness(m, n))
        row["T_theory_us"], row["bottleneck"] = (x * 1e6 if i == 0 else x
                                                 for i, x in enumerate(_bound(mach, w, row["dtype"])))
        row["eta"] = row["T_theory_us"] / row["T_meas_us"]
        row["T_tier_us"] = 6.0 * m * n * k / (TIER_PEAK_TFLOPS * 1e12 * TIER_GEMM_EFF * tier(m, n, k)) * 1e6
        row["tier"] = tier(m, n, k)
        row["meas_tflops"] = 6.0 * m * n * k / (row["T_meas_us"] * 1e-6) / 1e12
        rows.append(row)
    return rows


def non_gemm_rows(mach: TheoryMachine, profile: dict) -> list:
    """Lower bound only: torch-eager composites, so eta measures the implementation."""
    rows = []
    for key, v in profile.get("operators", {}).get("non_gemm", {}).items():
        kv = _key(key)
        row = {"key": key, "part": kv["part"], "elements": int(float(kv["elements"])),
               "T_meas_us": float(v["mean_ms"]) * 1e3}
        if kv["part"] in ("gated_delta_rule", "optimizer_step", "gdn_causal_conv1d"):
            row["excluded"] = "not elementwise (recurrence / optimizer / conv): needs its own plugin"
            rows.append(row)
            continue
        # Minimum for any unary fwd+bwd: read x, write y, read dy, read x, write dx.
        w = get("elementwise").work({"elements": row["elements"]}, kv.get("dtype", "bf16"), "fwd+bwd")
        w.kernels = 2
        t, b = _bound(mach, w, kv.get("dtype", "bf16"))
        row.update(T_theory_us=t * 1e6, bottleneck=b, eta=t * 1e6 / row["T_meas_us"])
        rows.append(row)
    return rows


def _err(pred: float, meas: float) -> float:
    return abs(pred - meas) / meas


def _spearman(a: list, b: list) -> float:
    def rank(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else 0.0


def fit_floor(xs: list, ys: list) -> tuple:
    """T_meas ~= a + b * T_theory minimising relative error (weights 1/T_meas^2).

    a is the per-call floor the measurement adds regardless of shape (framework dispatch,
    harness kernels, event timing); 1/b is the machine-wide efficiency on top of it.
    """
    w = [1.0 / (y * y) for y in ys]
    sw = sum(w)
    sx = sum(wi * x for wi, x in zip(w, xs))
    sy = sum(wi * y for wi, y in zip(w, ys))
    sxx = sum(wi * x * x for wi, x in zip(w, xs))
    sxy = sum(wi * x * y for wi, x, y in zip(w, xs, ys))
    den = sw * sxx - sx * sx
    b = (sw * sxy - sx * sy) / den
    a = (sy - b * sx) / sw
    return a, b


def summarize(rows: list) -> dict:
    ok = [r for r in rows if not r.get("excluded")]
    viol = [r for r in ok if r["T_theory_us"] > r["T_meas_us"] * 1.0001]
    etas = [r["eta"] for r in ok]
    out = {"entries": len(rows), "used": len(ok), "excluded": len(rows) - len(ok),
           "bound_violations": [r["key"] for r in viol],
           "eta_median": statistics.median(etas) if etas else None,
           "eta_min": min(etas) if etas else None, "eta_max": max(etas) if etas else None}
    if ok and "T_tier_us" in ok[0]:
        xs, ys = [r["T_theory_us"] for r in ok], [r["T_meas_us"] for r in ok]
        for i, r in enumerate(ok):
            r["T_eta_us"] = r["T_theory_us"] / statistics.median(etas[:i] + etas[i + 1:])
            a, b = fit_floor(xs[:i] + xs[i + 1:], ys[:i] + ys[i + 1:])
            r["T_pred_us"] = a + b * r["T_theory_us"]
        a, b = fit_floor(xs, ys)
        out["floor_fit_all"] = {"floor_us": a, "eta": 1.0 / b}

        def stats(col):
            e = [_err(r[col], r["T_meas_us"]) for r in ok]
            return {"mape_pct": 100 * statistics.mean(e), "median_ape_pct": 100 * statistics.median(e),
                    "max_ape_pct": 100 * max(e), "spearman": _spearman([r[col] for r in ok], ys)}
        out["predictor"] = {
            "hcu_tiered (4-step table + gemm_efficiency, fitted on this data)": stats("T_tier_us"),
            "kra: T_theory / eta (1 constant, leave-one-out)": stats("T_eta_us"),
            "kra: floor + T_theory / eta (2 constants, leave-one-out)": stats("T_pred_us"),
        }
        by = {}
        for r in ok:
            by.setdefault(r["bottleneck"], []).append(r["eta"])
        out["eta_by_bottleneck"] = {b: {"n": len(v), "median": statistics.median(v)}
                                    for b, v in by.items()}
    return out


def markdown(mach: TheoryMachine, gs: dict, grows: list, ns: dict, nrows: list) -> str:
    L = ["# kra validate: shape-only bounds vs BW1000 measurements (%s peaks)" % mach.which, "",
         "## GEMM (fwd + dX + dW per entry)", "",
         "entries %d, used %d, excluded %d; bound violations: %s" % (
             gs["entries"], gs["used"], gs["excluded"],
             ", ".join(gs["bound_violations"]) or "none"),
         "", "eta = T_theory / T_meas: median %.3f, range %.3f - %.3f" % (
             gs["eta_median"], gs["eta_min"], gs["eta_max"]), ""]
    if "predictor" in gs:
        L += ["fitted over all entries: per-call floor %.0f us, eta %.3f" % (
            gs["floor_fit_all"]["floor_us"], gs["floor_fit_all"]["eta"]), "","| predictor | MAPE | median APE | max APE | Spearman |", "| --- | ---: | ---: | ---: | ---: |"]
        for name, p in gs["predictor"].items():
            L.append("| %s | %.1f%% | %.1f%% | %.0f%% | %.3f |" % (
                name, p["mape_pct"], p["median_ape_pct"], p["max_ape_pct"], p["spearman"]))
        L += ["", "| bottleneck | n | median eta |", "| --- | ---: | ---: |"]
        for b, v in sorted(gs["eta_by_bottleneck"].items()):
            L.append("| %s | %d | %.3f |" % (b, v["n"], v["median"]))
    L += ["", "| part | M | N | K | meas us | meas TFLOPS | T_theory us | bottleneck | eta | kra pred us | tiered us |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |"]
    for r in sorted((r for r in grows if not r.get("excluded")), key=lambda r: r["eta"]):
        L.append("| %s | %d | %d | %d | %.1f | %.1f | %.1f | %s | %.3f | %.1f | %.1f |" % (
            r["part"], r["M"], r["N"], r["K"], r["T_meas_us"], r["meas_tflops"], r["T_theory_us"],
            r["bottleneck"], r["eta"], r["T_pred_us"], r["T_tier_us"]))
    L += ["", "excluded:", ""] + ["- `%s`: %s" % (r["key"], r["excluded"]) for r in grows if r.get("excluded")]
    L += ["", "## non_gemm (lower bound only)", "",
          "entries %d, used %d, excluded %d; bound violations: %s" % (
              ns["entries"], ns["used"], ns["excluded"], ", ".join(ns["bound_violations"]) or "none"),
          "", "eta median %.3f, range %.4f - %.3f: torch-eager composites run far from the HBM "
          "bound, which is the headroom a fused kernel would recover." % (
              ns["eta_median"], ns["eta_min"], ns["eta_max"]), ""]
    by = {}
    for r in nrows:
        if not r.get("excluded"):
            by.setdefault(r["part"], []).append(r["eta"])
    L += ["| part | n | median eta |", "| --- | ---: | ---: |"]
    for p, v in sorted(by.items(), key=lambda kv: statistics.median(kv[1])):
        L.append("| %s | %d | %.4f |" % (p, len(v), statistics.median(v)))
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="kra validate")
    p.add_argument("--machine", required=True)
    p.add_argument("--profile", required=True, help="hcu-train-simulator profile yaml")
    p.add_argument("--peaks", default="measured", choices=("measured", "theoretical"))
    p.add_argument("--no-harness", action="store_true",
                   help="bound the 3 GEMMs only, without the benchmark's cast/sum kernels")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    import yaml
    profile = yaml.safe_load(open(a.profile, encoding="utf-8"))
    mach = TheoryMachine.load(a.machine, a.peaks)
    grows = gemm_rows(mach, profile, harness=not a.no_harness)
    nrows = non_gemm_rows(mach, profile)
    gs, ns = summarize(grows), summarize(nrows)
    md = markdown(mach, gs, grows, ns, nrows)
    if a.out:
        open(a.out + ".validate.json", "w", encoding="utf-8").write(json.dumps(
            {"machine": mach.id, "peaks": mach.which, "gemm": {"summary": gs, "rows": grows},
             "non_gemm": {"summary": ns, "rows": nrows}}, indent=1) + "\n")
        open(a.out + ".validate.md", "w", encoding="utf-8").write(md)
        print("wrote %s.validate.json / .md" % a.out)
    else:
        print(md)
    return 1 if gs["bound_violations"] or ns["bound_violations"] else 0
