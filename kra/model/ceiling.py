"""Attainable ceiling: the best time a good implementation of this operator can reach on this machine,
and the arithmetic intensity that goes with it, level by level.

The chain the report has to keep separate, because each step has a different standing:

    T_theory  <=  T_split  <=  T_attain  <=  T_meas
       |            |            |
       |            |            +-- x O_idle: what this card loses to stalls nothing above models.
       |            |                NOT derivable from shapes.  It is a machine (and operator-family)
       |            |                calibration, read from the machine file section `attainable.idle`;
       |            |                where it is absent the ceiling IS the capacity bound and is reported
       |            |                as optimistic by that factor, never filled in with a guess.
       |            +-- x O_handoff = T_split / T_theory: the cost of the kernel split the on-chip
       |                capacity forces (handoff traffic through HBM, barriers, occupancy).  Derived.
       +-- the resource floor of the required work alone.

O_redund (the redundant work of one formulation against another) is not a coefficient here: the search
prices each formulation's own flops and bytes, so it shows up as the spread between formulations -- which
is the thing the user can act on, unlike a fitted number.

The intensity table is what makes the ceiling actionable.  Each memory level has its own required bytes,
so each has its own intensity and its own ridge point, and the attainable time is the largest level term.
A level whose intensity falls below its ridge point is what truncates the ceiling, and the gap between the
intensity the chosen split reaches and the best intensity any feasible split reaches at that level is
exactly the part of the ceiling the on-chip capacity takes away.
"""
from __future__ import annotations

LEVEL_PEAK = {"hbm": "hbm_read", "l2": "l2_read", "lds": "lds_read"}


def capability(mach) -> dict:
    """Which questions this machine file can answer (L0..L3), and what the next level is missing.

    The levels are not a taste ranking: each one is the set of fields a class of terms needs, so a report
    on an L1 file must not be read as if the concurrency or the dependency terms were in it."""
    tiers = [
        ("L0", "theoretical floor, intensity, ridge point, bottleneck side", [
            ("device.cu_count", bool(mach.cus)),
            ("HBM bandwidth", bool(mach.bw)),
            ("a compute peak", any(k.startswith("mmac") for k in mach.peaks))]),
        ("L1", "+ on-chip capacity: which kernel splits fit, what has to spill", [
            ("device.lds_per_cu", bool(mach.device.get("lds_per_cu") or mach.device.get("lds_per_block"))),
            ("device.regs_per_block", bool(mach.device.get("regs_per_block"))),
            ("device.max_threads_per_cu", bool(mach.device.get("max_threads_per_cu"))),
            ("device.l2_bytes", bool(mach.device.get("l2_bytes")))]),
        ("L2", "+ concurrency and latency: attainable bandwidth, exposed latency, launch, split ranking", [
            ("HBM bandwidth-vs-CTAs curve", bool(mach.bw_curves)),
            ("latency.gmem_hbm_ws", "gmem_hbm_ws" in mach.latency_s),
            ("launch overhead", mach.launch_us > 0)]),
        ("L3", "+ operator-dependent structure: attainable compute, memory issue, recurrence step", [
            ("structural.mmac_tile_eff", "mmac_tile_eff" in mach.structural),
            ("structural.vmem_cycles_per_kb", "vmem_cycles_per_kb" in mach.structural),
            ("structural.recurrence_step_us", "recurrence_step_us" in mach.structural)]),
    ]
    level, answers, missing = "none", "nothing: the floor itself cannot be solved", []
    for name, says, checks in tiers:
        gaps = [f for f, ok in checks if not ok]
        if gaps:
            missing = ["%s (for %s)" % (f, name) for f in gaps]
            break
        level, answers = name, says
    return {"level": level, "answers": answers, "missing_for_next": missing}


def idle_factor(mach, op_name: str) -> tuple:
    """(factor, provenance) from the machine file's `attainable.idle`, keyed by operator (then `_default`).

    A measured ratio of one implementation's time to this tool's capacity bound, on this card, for this
    family of operators.  It is a calibration: it does not transfer to another card or another family, so
    it is reported with its provenance and never extrapolated."""
    sec = (mach.raw.get("attainable") or {}).get("idle") or {}
    for key in (op_name, "_default"):
        v = sec.get(key)
        if isinstance(v, dict):
            f, src = v.get("value"), v.get("_provenance", "")
        else:
            f, src = v, (mach.raw.get("attainable") or {}).get("_provenance", "")
        if f:
            return float(f), src or "machine file attainable.idle, no provenance recorded"
    return None, None


def levels(flops: float, bytes_by_level: dict, best_bytes_by_level: dict, mach, peak: float,
           eff: float = 1.0) -> list:
    """One row per memory level: the bytes the split has to move through it, the intensity that gives, the
    level's ridge point, the time its roof allows, and the best intensity any feasible split reached."""
    out = []
    for name, key in LEVEL_PEAK.items():
        bw, b = mach.peaks.get(key), bytes_by_level.get(name)
        if not bw or not b:
            continue
        ceil_b = best_bytes_by_level.get(name) or b
        out.append({"level": name, "bw_GBps": bw / 1e9, "bytes": b,
                    "intensity_flop_per_byte": flops / b,
                    "intensity_ceiling_flop_per_byte": flops / ceil_b,
                    "ridge_flop_per_byte": peak * eff / bw,
                    "T_roof_us": b / bw * 1e6,
                    "bound": "compute" if flops / b > peak * eff / bw else "memory"})
    return sorted(out, key=lambda r: -r["T_roof_us"])


def report(mach, op_name: str, dtype: str, peak_key: str, peak: float, t_theory_us: float,
           best: dict, feasible_rows: list, formulations: list) -> dict:
    """The attainable ceiling of the best candidate, with every step of the chain attributed."""
    t_split = best["T_split_us"]
    eff = float(mach.structural.get("mmac_tile_eff", 1.0)) if "mmac_tile_eff" in mach.structural else 1.0
    best_bytes = {}
    for r in feasible_rows:
        for k, v in (r.get("bytes_levels") or {}).items():
            if v and (k not in best_bytes or v < best_bytes[k]):
                best_bytes[k] = v
    lv = levels(best["flops"], best.get("bytes_levels") or {}, best_bytes, mach, peak, eff)
    idle, src = idle_factor(mach, op_name)
    t_attain = t_split * (idle or 1.0)
    binding = max(best["per_kernel"], key=lambda k: k["T_us"])["binding"] if best.get("per_kernel") else ""
    spread = [f["T_split_us"] for f in formulations if "T_split_us" in f]
    notes = []
    if idle is None:
        notes.append("machine %s has no attainable.idle for %s: the attainable ceiling IS the "
                     "capacity-constrained bound, so it is optimistic by whatever this card loses to "
                     "stalls not modelled here (issue limits, cache misses, tail effects).  Calibrate it "
                     "with one measured kernel sequence of this operator family on this card."
                     % (mach.id, op_name))
    else:
        notes.append("attainable.idle = %.3f for %s on %s (%s): a calibration of this card and this "
                     "operator family, not a general coefficient." % (idle, op_name, mach.id, src))
    return {"capability": capability(mach),
            "T_theory_us": t_theory_us, "T_split_us": t_split, "T_attain_us": t_attain,
            "O_handoff": t_split / t_theory_us if t_theory_us else None,
            "O_idle": idle,
            "band_us": [t_split, t_attain],
            "formulation_spread_pct": (100.0 * (max(spread) / min(spread) - 1.0)) if len(spread) > 1 else 0.0,
            "binding": binding, "binding_level": lv[0]["level"] if lv else None,
            "levels": lv,
            "attainable_tflops": best["flops"] / (t_attain * 1e-6) / 1e12 if t_attain else None,
            "mfu_pct_at_attain": 100.0 * (best["flops"] / peak) / (t_attain * 1e-6) if t_attain else None,
            "peak_tflops": peak / 1e12, "compute_eff": eff, "peak_key": peak_key,
            "notes": notes}
