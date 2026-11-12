"""Shape-only prediction of an operator's performance ceiling.

Same bound structure as kra.model.bounds, but every term is computed from an operator
formula instead of from a measured window, so no run on hardware is needed:

  T_flop   required flops at the compute peak of the dtype
  T_mem    required bytes (inputs read once, outputs written once) at attainable HBM bw
  T_other  per-resource required work (e.g. transcendentals) at that resource's peak
  T_fixed  kernels x dispatch cost -- the floor small shapes hit

Per kernel the bound is max(all of them): every term is a lower bound on time, and the
largest binds.  Kernels that run back to back on one stream add.  The binding term of
the largest share is the bottleneck; the report also states each other resource's
utilisation at T_theory, which is what says how much a different split could recover.

T_theory is a *lower bound on time*, not an achievable runtime: see docs/sim/requirements-qa.md.
"""
from __future__ import annotations

import json

from ..ops import DTYPE_PEAK, Work, get

RESOURCE_UNITS = {"TFLOPS": 1e12, "TOPS": 1e12, "Gop/s": 1e9, "GB/s": 1e9}


class TheoryMachine:
    """Peaks and fixed costs of one device, in SI units per second.

    `peaks="theoretical"` uses the datasheet number where the machine file has one and
    falls back to the measured value otherwise; `peaks="measured"` uses the calibrated
    microbenchmark value.  Both are reported so a prediction always says which it used.
    """

    def __init__(self, m: dict, peaks: str = "measured"):
        self.raw = m
        self.which = peaks
        self.id = m.get("id") or m.get("device", {}).get("name", "?")
        self.device = m.get("device", {})
        self.peaks = {}
        for k, v in m.get("peaks", {}).items():
            if not isinstance(v, dict):
                continue
            val = v.get("theoretical") if peaks == "theoretical" else None
            if val is None:
                val = v.get("value")
            if val is None:
                continue
            unit = v.get("unit", "")
            scale = next((s for u, s in RESOURCE_UNITS.items() if unit.startswith(u)), None)
            if scale is None:
                continue
            self.peaks[k] = float(val) * scale
        self.bw = max((self.peaks[k] for k in ("hbm_read", "hbm_read_nt", "hbm_write", "hbm_copy")
                       if k in self.peaks), default=0.0)
        launch = m.get("launch", {}).get("launch_stream", {})
        self.launch_us = float(launch.get("best_us", 0.0))
        self.ridge = m.get("ridge_points_flop_per_byte", {})

    @classmethod
    def load(cls, path: str, peaks: str = "measured") -> "TheoryMachine":
        return cls(json.loads(open(path, encoding="utf-8").read()), peaks)

    def compute_peak(self, dtype: str) -> tuple:
        key = DTYPE_PEAK.get(dtype)
        if key is None:
            raise SystemExit("dtype %r has no compute peak mapping" % dtype)
        if key not in self.peaks:
            raise SystemExit(
                "machine %s has no %s peak, so %s cannot be predicted on it; add the peak "
                "to the machine file (measured or datasheet) first." % (self.id, key, dtype))
        return key, self.peaks[key]


def _terms(mach: TheoryMachine, w: Work, peak_key: str, peak: float) -> dict:
    t = {}
    if w.flops:
        t["compute:" + peak_key] = w.flops / peak
    if w.bytes_min:
        if not mach.bw:
            raise SystemExit("machine %s has no HBM bandwidth peak" % mach.id)
        t["memory:hbm"] = w.bytes_min / mach.bw
    for res, amount in w.other.items():
        if res in mach.peaks:
            t["other:" + res] = amount / mach.peaks[res]
    t["fixed:launch"] = w.kernels * mach.launch_us * 1e-6
    return t


def predict(mach: TheoryMachine, op_name: str, shape: dict, dtype: str, phase: str) -> dict:
    op = get(op_name)
    w: Work = op.work(shape, dtype, phase)
    peak_key, peak = mach.compute_peak(dtype)

    # A sequence of kernels on one stream: each is bounded by its own binding term and
    # the bounds add.  A single kernel is the one-part case.
    parts = w.parts or [(op_name, w)]
    rows, by_binding, terms = [], {}, {}
    for name, pw in parts:
        t = _terms(mach, pw, peak_key, peak)
        b = max(t, key=t.get)
        rows.append({"kernel": name, "binding": b, "T_us": t[b] * 1e6,
                     "terms_us": {k: v * 1e6 for k, v in t.items()}})
        by_binding[b] = by_binding.get(b, 0.0) + t[b]
        for k, v in t.items():
            terms[k] = terms.get(k, 0.0) + v
    t_theory = sum(by_binding.values())
    binding = max(by_binding, key=by_binding.get)
    oi = w.flops / w.bytes_min if w.bytes_min else float("inf")
    ridge = peak / mach.bw if mach.bw else float("inf")

    out = {
        "schema": "kra.predict/1",
        "machine": {"id": mach.id, "peaks": mach.which,
                    "compute_peak_tflops": peak / 1e12, "hbm_bw_GBps": mach.bw / 1e9,
                    "launch_us": mach.launch_us},
        "op": {"name": op_name, "phase": phase, "dtype": dtype, "shape": shape,
               "kernels": w.kernels},
        "work": {"flops": w.flops, "bytes_min": w.bytes_min, "working_set": w.working_set,
                 "other": w.other},
        "arithmetic_intensity_flop_per_byte": oi,
        "ridge_point_flop_per_byte": ridge,
        "terms_us": {k: v * 1e6 for k, v in terms.items()},
        "kernels": rows,
        "binding_share": {k: v / t_theory for k, v in by_binding.items()} if t_theory else {},
        "binding": binding,
        "bottleneck": binding.split(":")[0],
        "T_theory_us": t_theory * 1e6,
        "ceiling": {
            "mfu_pct": 100.0 * (w.flops / peak) / t_theory if t_theory else 0.0,
            "bw_util_pct": 100.0 * (w.bytes_min / mach.bw) / t_theory if (t_theory and mach.bw) else 0.0,
        },
        "notes": list(w.notes),
        "disclaimer": "T_theory is a lower bound on time under this machine's peaks; it is "
                      "not a runtime any implementation is guaranteed to reach.",
    }
    out["notes"].insert(0, _why(out))
    return out


def _why(r: dict) -> str:
    oi, ridge = r["arithmetic_intensity_flop_per_byte"], r["ridge_point_flop_per_byte"]
    side = "above" if oi > ridge else "below"
    return ("arithmetic intensity %.1f flop/byte is %s the ridge point %.1f, and %s binds "
            "at %.2f us" % (oi, side, ridge, r["binding"], r["T_theory_us"]))


def markdown(r: dict) -> str:
    o, m = r["op"], r["machine"]
    L = ["# %s %s on %s (%s peaks)" % (o["name"], o["phase"], m["id"], m["peaks"]), "",
         "shape: `%s`  dtype: `%s`  kernels: %d" % (
             ",".join("%s=%s" % kv for kv in o["shape"].items()), o["dtype"], o["kernels"]), "",
         "| quantity | value |", "| --- | ---: |",
         "| required flops | %.3f GFLOP |" % (r["work"]["flops"] / 1e9),
         "| required HBM bytes | %.3f MB |" % (r["work"]["bytes_min"] / 1e6),
         "| arithmetic intensity | %.1f flop/byte |" % r["arithmetic_intensity_flop_per_byte"],
         "| ridge point | %.1f flop/byte |" % r["ridge_point_flop_per_byte"],
         "| **T_theory** | **%.3f us** |" % r["T_theory_us"],
         "| bottleneck | %s (%s) |" % (r["bottleneck"], r["binding"]),
         "| ceiling MFU | %.1f%% |" % r["ceiling"]["mfu_pct"],
         "| ceiling HBM utilisation | %.1f%% |" % r["ceiling"]["bw_util_pct"], "",
         "## per-kernel bound", "", "| kernel | binding | T_us | " +
         " | ".join(sorted(r["terms_us"])) + " |",
         "| --- | --- | ---: |" + " ---: |" * len(r["terms_us"])]
    for row in r["kernels"]:
        L.append("| %s | %s | %.3f | " % (row["kernel"], row["binding"], row["T_us"]) +
                 " | ".join("%.3f" % row["terms_us"].get(k, 0.0) for k in sorted(r["terms_us"])) + " |")
    L += ["", "## notes", ""] + ["- " + n for n in r["notes"]]
    L += ["", "> " + r["disclaimer"], ""]
    return "\n".join(L)
