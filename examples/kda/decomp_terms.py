"""Calibrated model terms for decomp_eval.py blind spots
(docs/tasks/kda_decomp_calib_task.md).

Constants all trace to measurements; provenance is in
results/kda-decomp-calib-20261013/calib.json `note` fields and README.md.

    lam_scale(kernel, res_old, res_new, shape) -> float
    thr_cap(kernel, res_new, shape) -> float
    loop_inflation(kernel, d_live_vgpr) -> float
    icache_factor(code_bytes, n_paths, ctas_per_cu) -> float
    l2_extra_hbm_bytes(kernel, shape, ctas_resident, ...) -> float
"""
import json
from pathlib import Path

_CAL = json.load(open(Path(__file__).parent.parent.parent /
                      "results" / "kda-decomp-calib-20261013" / "calib.json"))

P1 = _CAL["p1"]
P2 = _CAL["p2"]
P3 = _CAL["p3"]
P4 = _CAL["p4"]


# ---- P1: residency -> latency ------------------------------------------------
def lam_scale(kernel_name: str, res_old: int, res_new: int, shape=None) -> float:
    """Time multiplier when a kernel's residency changes res_old -> res_new.
    Measured t(res) ratio of the kernel's class; outside the measured range the
    shallowest measured slope is used (down-sweep only: residency can only be
    lowered by the LDS topup probe)."""
    base = kernel_name.split("#")[0]
    cls = P1["class_of"].get(base, "generic")
    curve = P1["curves"].get(cls) or P1["curves"]["generic"]

    def val(r):
        v = curve.get(str(r))
        if not isinstance(v, dict):
            return v
        if shape and str(shape) in v:                   # shape-level dict -> kernel level
            v = v[str(shape)]
        if isinstance(v, dict):
            kk = "s4a" if "s4a" in base else "s4b" if "s4b" in base else \
                 "a" if base.endswith("_a") else "b" if base.endswith("_b") else base
            if kk in v:
                return v[kk]
            flat = []
            for x in v.values():
                flat += x.values() if isinstance(x, dict) else [x]
            return sum(flat) / len(flat) if flat else None
        return v

    t0, t1 = val(res_old), val(res_new)
    if t0 is not None and t1 is not None:
        return t1 / t0
    # unmeasured direction: residency INCREASE (e.g. s4b 2->3) uses the calibrated
    # conservative bound (dav-class marginal slope), not the steep 1->2 edge.
    ex = P1.get("extrap_2to3", {})
    if res_new > res_old and ex:
        per_step = ex.get(cls, ex.get("generic", 0.90))
        return per_step ** (res_new - res_old)
    ks = sorted(int(k) for k in curve)
    if len(ks) >= 2:
        slope = (curve[str(ks[-1])] / curve[str(ks[0])]) ** (1 / (ks[-1] - ks[0]))
        return slope ** (res_new - res_old)
    return 1.0


def thr_cap(kernel_name: str, res_new: int, shape=None) -> float:
    return P1["thr_cap_of"].get(kernel_name.split("#")[0], {}).get(str(res_new), 1.0)


# ---- P2: register pressure -> main loop inflation ----------------------------
def loop_inflation(kernel_name: str, d_live_vgpr: int) -> float:
    """Main-loop instruction-time multiplier when d_live_vgpr registers stay
    live across the kernel body (P-55 mechanism).  Piecewise: free headroom to
    the 256 cap, then the measured per-VGPR slope, plus a spill surcharge once
    scratch appears."""
    k = P2["per_kernel"].get(kernel_name.split("#")[0])
    if not k:
        slope = P2["inst_per_vgpr"]["generic"]
        return 1.0 + slope * max(d_live_vgpr, 0)
    headroom = max(k["vgpr0"] + d_live_vgpr - 256, 0)   # spilled part
    nospill = d_live_vgpr - headroom
    return (1.0 + k["slope_nospill"] * nospill) * (1.0 + k["slope_spill"] * headroom)


# ---- P3: code size -> icache pressure ----------------------------------------
def icache_factor(code_bytes: int, n_paths: int = 4, ctas_per_cu: int = 1) -> float:
    """cycles/instruction multiplier vs total kernel code footprint.
    Piecewise fit of the synthetic 4-path probe (P3 figure); co-resident CTAs
    multiply the footprint."""
    # same code is shared by all CTAs of a kernel; co-residency only matters for
    # DISTINCT kernels.  ctas_per_cu kept for signature compatibility.
    fp = code_bytes
    for cap, mult in P3["curve"]:
        if fp <= cap:
            return mult
    return P3["curve"][-1][1]


# ---- P4: L2 capacity -> extra HBM bytes --------------------------------------
def l2_extra_hbm_bytes(kernel_name: str, shape, ctas_resident: int = None,
                       req_rd: float = 0.0, variant: str = "narrow") -> float:
    """Extra HBM READ bytes for the fwd recurrence when the shared K-side
    working set outlives L2 (thrash).  Calibrated on cards (extra/req_rd ~ a*T +
    b*H + g for the narrow kernel; the wide variant re-reads ~0 at H<=48).
    variant: 'narrow' (v8o4, 4 CTAs/head) or 'wide' (v8o4w, 2 CTAs/head)."""
    if variant != "narrow":
        return 0.0
    T, H = map(int, shape.split("_"))
    c = P4["extra_frac_coef"]
    frac = c["alpha_T"] * T / 1024 + c["beta_H"] * H + c["gamma"]
    return req_rd * max(0.0, min(frac, c["cap"]))


def l2_lam_factor(kernel_name: str, shape, variant: str = "narrow") -> float:
    """Residence-time multiplier on the fwd recurrence when the counterfactual
    working set thrashes L2.  1 + kappa * max(0, extra_linear - 1.0)."""
    if variant != "narrow":
        return 1.0
    T, H = map(int, shape.split("_"))
    c = P4["extra_frac_coef"]
    extra_lin = c["alpha_T"] * T / 1024 + c["beta_H"] * H + c["gamma"]
    return 1.0 + P4["lam_kappa"] * max(0.0, extra_lin - 1.0)
