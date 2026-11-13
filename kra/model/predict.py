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

    def __init__(self, m: dict, peaks: str = "measured", occupancy: bool = True):
        self.raw = m
        self.use_occupancy = occupancy
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
        self.cus = int(self.device.get("cu_count", 0))
        self.clk = float(self.device.get("clock_khz", 0)) * 1e3
        if self.cus and self.clk:
            # one lane-op per lane per cycle; packed fp32 pairs are the plugin's business
            self.peaks.setdefault("valu_lane", self.cus * int(self.device.get("warp_size", 64)) * self.clk)
        self.structural = {k: v for k, v in m.get("structural", {}).items() if not k.startswith("_")}
        self.latency_s = {k: float(v["ns"]) * 1e-9 for k, v in m.get("latency", {}).items()
                          if isinstance(v, dict) and "ns" in v}
        # measured attainable bandwidth as a function of how many CTAs are resident (streaming
        # microbenchmark): the ceiling a kernel with that much parallelism can reach.
        self.bw_curves = {k: {int(c): float(b) for c, b in v["curve"].items()}
                          for k, v in m.get("peaks", {}).items()
                          if isinstance(v, dict) and isinstance(v.get("curve"), dict)}

    def bw_at_ctas(self, key: str, ctas: int) -> float | None:
        """Attainable bandwidth of `key` with `ctas` CTAs resident, interpolated on the measured
        curve; None when the machine file has no curve for it."""
        cur = self.bw_curves.get(key)
        if not cur:
            return None
        xs = sorted(cur)
        if ctas >= xs[-1]:
            return cur[xs[-1]] * 1e9
        if ctas <= xs[0]:
            return cur[xs[0]] * 1e9 * ctas / xs[0]
        hi = next(x for x in xs if x >= ctas)
        lo = max(x for x in xs if x <= ctas)
        f = 0.0 if hi == lo else (ctas - lo) / (hi - lo)
        return (cur[lo] + f * (cur[hi] - cur[lo])) * 1e9

    def concurrency(self, occ: dict) -> dict:
        """What one kernel can keep in flight on this machine, from the hardware limits only.

        Resident waves per CU is the smallest of what LDS, the register file, the CU's wave slots and
        the grid allow.  Whatever of the register file those waves do not hold live is the deepest
        prefetch they could run, which is an *upper* bound on the bytes in flight -- so the latency it
        gives is a lower bound on time.  Register-hungry, low-residency kernels get a smaller one."""
        dev = self.device
        warp = int(dev.get("warp_size", 64) or 64)
        thr = int(occ.get("threads") or warp)
        per_cta = max(1, thr // warp)
        regs_per_cu = float(dev.get("regs_per_block") or 0)      # dwords of the CU register file
        lds_per_cu = float(dev.get("lds_per_cu") or dev.get("lds_per_block") or 0)
        vgpr = max(1.0, float(occ.get("vgpr") or 1.0))
        lim, why = {}, []
        if lds_per_cu and occ.get("lds_bytes"):
            lim["lds"] = int(lds_per_cu // occ["lds_bytes"]) * per_cta
        if regs_per_cu:
            lim["vgpr"] = int(regs_per_cu // (vgpr * warp))
        if dev.get("max_threads_per_cu"):
            lim["slots"] = int(float(dev["max_threads_per_cu"]) // warp)
        if self.cus and occ.get("ctas"):
            lim["grid"] = max(1, -(-int(occ["ctas"]) // self.cus)) * per_cta
        if not lim or not regs_per_cu:
            return {}
        waves = max(1, min(lim.values()))
        binding = min(lim, key=lim.get)
        # Deepest prefetch those waves could run: what the register file leaves them, and never more
        # than the architectural per-thread register limit a single wave can address.
        free = min(regs_per_cu / (warp * waves), float(dev.get("max_vgpr_per_thread") or 256)) - vgpr
        if free < 1.0:
            free, why = 1.0, ["registers are fully live: one dword per lane assumed in flight"]
        return {"waves_per_cu": waves, "occupancy_limit": binding, "ctas_per_cu": waves / per_cta,
                "wave_slots_pct": 100.0 * waves / max(1.0, lim.get("slots", waves)),
                "cus_used": int(min(int(occ.get("ctas") or 0) or self.cus, self.cus)) if self.cus else 0,
                "ctas_resident": int(min(int(occ.get("ctas") or 0) or 10 ** 9,
                                         self.cus * max(1, waves // per_cta))),
                "in_flight_bytes_per_cu": waves * warp * free * 4.0,
                "free_dwords_per_lane": free, "notes": why}

    def step_s(self, key: str, ctas: int) -> float | None:
        """Latency of one dependent recurrence step, or None if the machine file lacks it."""
        tab = self.structural.get("recurrence_step_us", {}).get(key)
        if not tab:
            return None
        per_cu = max(1, -(-ctas // self.cus)) if self.cus else 1
        return float(tab.get("%dcu" % per_cu, tab[max(tab)])) * 1e-6

    @classmethod
    def load(cls, path: str, peaks: str = "measured", occupancy: bool = True) -> "TheoryMachine":
        return cls(json.loads(open(path, encoding="utf-8").read()), peaks, occupancy)

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
    con = mach.concurrency(w.occupancy) if (w.occupancy and mach.use_occupancy) else {}
    # A kernel with fewer tiles of work than the device has CUs can only occupy that many CUs, so the
    # per-CU resources (matrix and vector pipes, LDS, memory issue) are available to it in that
    # fraction only.  Device-level resources (HBM, L2) are not scaled this way; HBM gets the measured
    # bandwidth-vs-CTAs curve instead.
    cu_frac = min(1.0, con["cus_used"] / mach.cus) if con.get("cus_used") and mach.cus else 1.0
    if w.flops:
        eff = float(mach.structural.get(w.compute_eff, 1.0)) if w.compute_eff else 1.0
        t["compute:" + peak_key] = w.flops / (peak * eff * cu_frac)
    if w.bytes_min or w.bytes_rd:
        if not mach.bw:
            raise SystemExit("machine %s has no HBM bandwidth peak" % mach.id)
        if w.bytes_rd is not None and "hbm_read" in mach.peaks and "hbm_write" in mach.peaks:
            # A kernel with too few resident CTAs cannot reach the streaming peak even in principle:
            # charge its reads at the bandwidth measured at that CTA count.  Writes are left at the
            # peak, because no write-vs-CTA curve was measured -- under-charging keeps the bound below.
            bw_rd = mach.peaks["hbm_read"]
            if con.get("ctas_resident"):
                at = mach.bw_at_ctas("hbm_read_vs_ctas", con["ctas_resident"])
                if at:
                    bw_rd = min(bw_rd, at)
            t["memory:hbm"] = w.bytes_rd / bw_rd + (w.bytes_wr or 0.0) / mach.peaks["hbm_write"]
        else:
            t["memory:hbm"] = w.bytes_min / mach.bw
        # multi-level roofline: the same required-bytes argument one level in.  Every HBM byte
        # also crosses L2, so bytes_l2 defaults to the HBM traffic; LDS traffic has to be declared
        # by the operator (staged tiles) or the level is left out.
        lvl = {"l2": (w.bytes_l2 if w.bytes_l2 is not None else w.bytes_min, "l2_read"),
               "lds": (w.bytes_lds, "lds_read")}
        for name, (b, key) in lvl.items():
            if b and key in mach.peaks:
                t["memory:" + name] = b / (mach.peaks[key] * (cu_frac if name == "lds" else 1.0))
        vm = mach.structural.get("vmem_cycles_per_kb")
        if w.vmem_issue and vm and mach.cus and w.bytes_rd is not None:
            t["issue:vmem"] = (w.bytes_rd / 1024 * vm["read"] + (w.bytes_wr or 0.0) / 1024 * vm["write"]) \
                / (mach.cus * cu_frac * mach.clk)
    if con.get("in_flight_bytes_per_cu") and mach.cus and w.bytes_rd and mach.latency_s.get("gmem_hbm_ws"):
        # Little's law the other way round: the bytes a CU must read, at the rate its own in-flight
        # window and the measured memory latency allow.  Independent of bandwidth: it is what the
        # kernel's own parallelism cannot hide.
        t["latency:hbm"] = (w.bytes_rd / mach.cus) * mach.latency_s["gmem_hbm_ws"] \
            / con["in_flight_bytes_per_cu"]
    for res, amount in w.other.items():
        if res in mach.peaks:
            t["other:" + res] = amount / (mach.peaks[res] * cu_frac)
    if w.serial:
        ch, missing = 0.0, []
        for steps, key, ctas in w.serial:
            s = mach.step_s(key, ctas)
            if s is None:
                missing.append(key)
            else:
                ch += steps * s
        if missing:
            w.notes.append("machine %s has no recurrence_step_us for %s: the serial chain is NOT in "
                           "this bound" % (mach.id, ",".join(sorted(set(missing)))))
        if ch:
            t["serial:chain"] = ch
    t["fixed:launch"] = w.kernels * mach.launch_us * 1e-6
    return t


def _bound(mach: TheoryMachine, parts: list, peak_key: str, peak: float) -> tuple:
    """Kernels back to back on one stream: each bounded by its own binding term, summed."""
    rows, by_binding, terms = [], {}, {}
    for name, pw in parts:
        t = _terms(mach, pw, peak_key, peak)
        b = max(t, key=t.get)
        rows.append({"kernel": name, "binding": b, "T_us": t[b] * 1e6,
                     "terms_us": {k: v * 1e6 for k, v in t.items()}})
        by_binding[b] = by_binding.get(b, 0.0) + t[b]
        for k, v in t.items():
            terms[k] = terms.get(k, 0.0) + v
    return sum(by_binding.values()), rows, by_binding, terms


def predict(mach: TheoryMachine, op_name: str, shape: dict, dtype: str, phase: str) -> dict:
    op = get(op_name)
    w: Work = op.work(shape, dtype, phase)
    peak_key, peak = mach.compute_peak(dtype)

    # A sequence of kernels on one stream: each is bounded by its own binding term and
    # the bounds add.  A single kernel is the one-part case.
    t_theory, rows, by_binding, terms = _bound(mach, w.parts or [(op_name, w)], peak_key, peak)
    splits = {}
    for sname, sparts in w.splits.items():
        ts, srows, sbb, _ = _bound(mach, sparts, peak_key, peak)
        for _, pw in sparts:
            w.notes += [n for n in pw.notes if n not in w.notes]
        splits[sname] = {"T_split_us": ts * 1e6, "O_handoff": ts / t_theory if t_theory else None,
                         "kernels": srows,
                         "bytes": sum((pw.bytes_rd or 0) + (pw.bytes_wr or 0) or pw.bytes_min for _, pw in sparts),
                         "binding_share": {k: v / ts for k, v in sbb.items()} if ts else {}}
    binding = max(by_binding, key=by_binding.get)
    oi = w.flops / w.bytes_min if w.bytes_min else float("inf")
    ridge = peak / mach.bw if mach.bw else float("inf")
    # hierarchical roofline: where the operator sits against each level's roof.  The traffic of a
    # level the operator does not bound is left out rather than guessed.
    levels = {}
    for name, key, b in (("hbm", "hbm_read", w.bytes_min),
                         ("l2", "l2_read", w.bytes_l2 if w.bytes_l2 is not None else w.bytes_min),
                         ("lds", "lds_read", w.bytes_lds)):
        bw = mach.peaks.get(key)
        if not bw or not b:
            continue
        levels[name] = {"bw_GBps": bw / 1e9, "bytes": b, "intensity_flop_per_byte": w.flops / b,
                        "ridge_flop_per_byte": peak / bw, "T_us": b / bw * 1e6,
                        "bound": "compute" if w.flops / b > peak / bw else "memory"}

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
        "levels": levels,
        "ridge_point_flop_per_byte": ridge,
        "terms_us": {k: v * 1e6 for k, v in terms.items()},
        "kernels": rows,
        "binding_share": {k: v / t_theory for k, v in by_binding.items()} if t_theory else {},
        "binding": binding,
        "bottleneck": binding.split(":")[0],
        "T_theory_us": t_theory * 1e6,
        "T_split": splits,
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
         "## hierarchical roofline", "",
         "| level | attainable bw | required bytes | intensity | ridge | bytes/bw | side |",
         "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for name, lv in r.get("levels", {}).items():
        L.append("| %s | %.0f GB/s | %.1f MB | %.1f | %.1f | %.1f us | %s |" % (
            name, lv["bw_GBps"], lv["bytes"] / 1e6, lv["intensity_flop_per_byte"],
            lv["ridge_flop_per_byte"], lv["T_us"], lv["bound"]))
    L += ["",
         "## per-kernel bound", "", "| kernel | binding | T_us | " +
         " | ".join(sorted(r["terms_us"])) + " |",
         "| --- | --- | ---: |" + " ---: |" * len(r["terms_us"])]
    for row in r["kernels"]:
        L.append("| %s | %s | %.3f | " % (row["kernel"], row["binding"], row["T_us"]) +
                 " | ".join("%.3f" % row["terms_us"].get(k, 0.0) for k in sorted(r["terms_us"])) + " |")
    for sname, sp in r.get("T_split", {}).items():
        L += ["", "## T_split: decomposition `%s` -- %.3f us (O_handoff = T_split / T_theory = %.2f, "
              "interface bytes %.1f MB = %.2fx required)" % (
                  sname, sp["T_split_us"], sp["O_handoff"], sp["bytes"] / 1e6,
                  sp["bytes"] / max(1.0, r["work"]["bytes_min"])), "",
              "| kernel | binding | T_us |", "| --- | --- | ---: |"]
        L += ["| %s | %s | %.1f |" % (k["kernel"], k["binding"], k["T_us"]) for k in sp["kernels"]]
    L += ["", "## notes", ""] + ["- " + n for n in r["notes"]]
    L += ["", "> " + r["disclaimer"], ""]
    return "\n".join(L)
