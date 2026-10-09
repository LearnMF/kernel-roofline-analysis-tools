"""L2 measured classification + L4 lower bounds, per kernel and per launcher call.

Inputs: machine.json (L0), pmc.json (L2 counters of one measured window),
op_spec.json (launcher -> kernels, serial structure), optrace.json (interface bytes per
launcher call, from kra.opspec.capture).

Lower-bound components (all in us, all optimistic by construction):
  T_mem_min   interface bytes (inputs read once, outputs written once) at attainable HBM
              read/write bandwidth                                         [prior]
  T_mem_impl  measured HBM bytes at attainable bandwidth                    [impl]
  T_par_*     same, but with the bandwidth attainable by the kernel's number of CTAs
              (machine.peaks.hbm_read_vs_ctas) and VALU-pipe slots spread over the
              active CUs only (grid < CU count)
  T_pipe      VALU-pipe issue slots (incl. MMAC 2 slots, transcendental 4) at 1 slot per
              CU per cycle, i.e. 100% pipe utilization                     [impl]
  T_cp_min    serial steps x dependency-only chain (op_spec.serial.chain_min)  [prior]
  T_cp_impl   serial steps x current-dataflow chain (op_spec.serial.chain)    [impl]
  T_launch    kernels x back-to-back dispatch cost (L0)
T_lower_prior = max(prior components); T_lower_impl = max(impl components).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..timeline.analyze import short_name

_THRESH = Path(__file__).resolve().parent.parent / "thresholds.json"


def thresholds() -> dict:
    return {k: v["value"] for k, v in json.loads(_THRESH.read_text())["thresholds"].items()}


class Machine:
    def __init__(self, m: dict):
        self.raw = m
        d, p, lat = m["device"], m["peaks"], m["latency"]
        self.cus = d["cu_count"]
        self.clk = d["clock_khz"] * 1e3
        # One bandwidth for all bytes: the highest measured HBM rate.  Real kernels with
        # mixed read+write traffic exceed the write/copy microbenchmarks (KDA l2n_apply:
        # 1271 GB/s mixed vs copy 1146), so separate read/write peaks would not be a bound.
        self.bw = max(p[k]["value"] for k in ("hbm_read", "hbm_read_nt", "hbm_write", "hbm_copy")
                      if k in p) * 1e9
        self.curve = {int(k): v * 1e9 for k, v in (p.get("hbm_read_vs_ctas") or {}).get("curve", {}).items()}
        self.mmac_bf16 = p["mmac_bf16"]["value"] * 1e12
        self.lat = {"lds": lat["lds"]["cycles"], "mmac_dep": lat["mmac_bf16_dependent"]["cycles"],
                    "gmem_hbm": lat["gmem_hbm_ws"]["cycles"], "gmem_l2": lat["gmem_l2_ws"]["cycles"]}
        self.barrier = {int(k.split("_")[-1]): v["cycles"] for k, v in lat["barrier"].items()
                        if k.startswith("lds_barrier")}
        self.launch_us = m["launch"]["launch_stream"]["best_us"]

    def bw_for(self, ctas: int) -> float:
        """Attainable HBM bandwidth for a grid of `ctas` CTAs (piecewise linear in the
        measured CTA curve, capped at the overall peak)."""
        if ctas >= self.cus or not self.curve:
            return self.bw
        xs = sorted(self.curve)
        if ctas <= xs[0]:
            return min(self.bw, self.curve[xs[0]] * ctas / xs[0])
        for a, b in zip(xs, xs[1:]):
            if a <= ctas <= b:
                return min(self.bw, self.curve[a] + (self.curve[b] - self.curve[a]) * (ctas - a) / (b - a))
        return self.bw

    def barrier_cycles(self, block: int) -> float:
        if not self.barrier:
            return 0.0
        ks = sorted(self.barrier)
        best = min(ks, key=lambda k: (abs(k - block), k))
        return self.barrier[best]

    def chain_cycles(self, chain: dict, block: int) -> float:
        prim = {"barrier": self.barrier_cycles(block), **self.lat}
        unknown = set(chain) - set(prim)
        if unknown:
            raise ValueError(f"unknown chain primitives {unknown}; known: {sorted(prim)}")
        return sum(n * prim[k] for k, n in chain.items())


def _eval(expr, env):
    if isinstance(expr, (int, float)):
        return expr
    return eval(expr, {"__builtins__": {}}, dict(env))  # op_spec formulas: arithmetic only


def kernel_metrics(k: dict, M: Machine, th: dict) -> dict:
    t = k["t_us"] * 1e-6
    rd, wr = k["hbm_rd_bytes"] or 0.0, k["hbm_wr_bytes"] or 0.0
    active = min(k["ctas"], M.cus) if k["ctas"] else M.cus
    bw_par = M.bw_for(k["ctas"])
    slots, insts = k["valu_slots"], k["valu_insts"]
    mmac_est = max(0.0, slots - insts)
    T = {
        "mem_impl": (rd + wr) / M.bw * 1e6,
        "par_mem_impl": (rd + wr) / bw_par * 1e6,
        "pipe": slots / (M.cus * M.clk) * 1e6,
        "par_pipe": slots / (active * M.clk) * 1e6,
        "mmac_impl": mmac_est * 8192 / M.mmac_bf16 * 1e6,
        "launch": M.launch_us,
    }
    # Utilization denominators use the faster of the two replays (t_us = min): replays of
    # long compute kernels differ by up to ~12% on gfx936 (pmccal), and GRBM cycles come
    # from the read replay only.
    cyc = t * M.clk if t else k["cycles"]
    sol_pipe = slots / (cyc * M.cus) if cyc else 0.0
    sol_mem = T["mem_impl"] / k["t_us"] if k["t_us"] else 0.0
    m = {
        "t_us": k["t_us"], "ctas": k["ctas"], "block": k["block"], "active_cus": active,
        "hbm_GBps": (rd + wr) / t / 1e9 if t else 0.0, "hbm_rd_MB": rd / 1e6, "hbm_wr_MB": wr / 1e6,
        "sol_pipe": sol_pipe, "sol_mem": sol_mem,
        "sol_pipe_active": slots / (cyc * active) if cyc else 0.0,
        "sol_mem_active": T["par_mem_impl"] / k["t_us"] if k["t_us"] else 0.0,
        "mmac_slot_share": 2 * mmac_est / slots if slots else 0.0,
        "lds_conflict_ratio": k["lds_bank_conflict_cycles"] / (cyc * M.cus) if cyc else 0.0,
        # SQ_WAIT_INST_LDS: wave-cycles (units of 4) waiting to issue LDS instructions, per
        # SIMD-cycle.  Criticality evidence for LDS findings (validation P-1: a 19.6% conflict
        # ratio with a 3.6% LDS wait share was real but off the critical path, 1.4% gain).
        "lds_wait_share": 4 * k.get("lds_wait", 0.0) / (cyc * M.cus * 4) if cyc else 0.0,
        "vgpr": k["arch_vgpr"] + k["accum_vgpr"], "lds_bytes": k["lds_bytes"], "scratch": k["scratch"],
        "T": T,
    }
    return m


def classify(m: dict, serial: bool, th: dict, cus: int) -> dict:
    lo, hi = th["kernel.sol_low"], th["kernel.sol_high"]
    b_lo, b_hi = th["kernel.balanced_band"]
    sp, sm = m["sol_pipe"], m["sol_mem"]
    reasons = [f"pipe {sp:.0%}, HBM {sm:.0%} of attainable"]
    if max(sp, sm) >= lo:
        if sp >= sm:
            cls = "B.compute.mmac" if m["mmac_slot_share"] >= 0.5 else "B.compute.valu"
        else:
            cls = "B.memory.hbm"
        sub = []
        level = "near-limit" if max(sp, sm) >= hi else "throughput"
    elif b_lo <= sp < b_hi and b_lo <= sm < b_hi:
        cls, sub, level = "E.balanced", [], "balanced"
    else:
        cls, level, sub = "C", "latency", []
        if m["ctas"] and m["ctas"] < cus:
            sub.append("C.parallelism")
            reasons.append(f"{m['ctas']} CTAs < {cus} CUs; on active CUs: pipe {m['sol_pipe_active']:.0%}, "
                           f"HBM {m['sol_mem_active']:.0%}")
        if serial:
            sub.append("C.serial")
        if not sub:
            sub.append("C.unresolved")
            reasons.append("latency subtype (mem latency / sync / divergence) needs L3 (SQTT)")
    causes = []
    if m["lds_conflict_ratio"] >= 0.05:
        critical = m.get("lds_wait_share", 1.0) >= th.get("kernel.lds_wait_critical", 0.05)
        causes.append({"id": "D.lds_conflict", "value": round(m["lds_conflict_ratio"], 3),
                       "critical": critical, "lds_wait_share": round(m.get("lds_wait_share", 0.0), 3),
                       "note": "on the critical path: removing it should save time" if critical else
                               "present but NOT critical (LDS issue wait small): expected time gain small"})
    if m["scratch"]:
        causes.append({"id": "D.spill", "value": m["scratch"]})
    if m["t_us"] is not None and m["t_us"] < th["system.small_kernel_us"]:
        causes.append({"id": "E.tiny", "value": m["t_us"]})
    return {"class": cls, "subtypes": sub, "regime": level, "causes": causes, "reasons": reasons}


def align(calls: list[dict], groups: dict, pmc: list[dict]) -> list[dict]:
    """Assign PMC kernels (in order) to launcher calls (in order) using op_spec regexes.
    Kernels not claimed by any launcher become single-kernel 'unattributed' entries."""
    out, j = [], 0
    for c in calls:
        g = groups.get(c["launcher"])
        if g is None:
            continue
        ks = []
        for pat in g["kernels"]:
            optional = pat.endswith("?")
            rx = re.compile(pat.rstrip("?"))
            if optional:
                if j < len(pmc) and rx.search(pmc[j]["name"]):
                    ks.append(pmc[j]); j += 1
                continue
            while j < len(pmc) and not rx.search(pmc[j]["name"]):
                out.append({"launcher": None, "call": None, "spec": None, "kernels": [pmc[j]]})
                j += 1
            if j >= len(pmc):
                raise ValueError(f"kernel /{pat}/ of launcher {c['launcher']} not found in PMC window")
            ks.append(pmc[j]); j += 1
        out.append({"launcher": c["launcher"], "call": c, "spec": g, "kernels": ks})
    for k in pmc[j:]:
        out.append({"launcher": None, "call": None, "spec": None, "kernels": [k]})
    return out


def verdict(att_prior: float, att_impl: float, dominant_class: str, th: dict) -> str:
    if att_prior > 1.0 + 1e-6:
        return "bound_violated"
    if att_prior >= th["stop.near_ceiling"]:
        return "at_ceiling"
    if att_impl >= th["stop.near_ceiling"]:
        return "hardware_busy_redundant_work"
    if dominant_class.startswith("B."):
        return "throughput_bound"
    if att_prior < th["stop.radical_below"]:
        return "large_headroom"
    return "moderate_headroom"


VERDICT_TEXT = {
    "at_ceiling": "已接近该数据流在本硬件上的下限：继续微调收益有限，需改算法或换硬件",
    "hardware_busy_redundant_work": "硬件单元已忙，但做了多余的访存/计算：减少流量或工作量（融合、布局、复用）",
    "throughput_bound": "主要 kernel 受某个硬件单元吞吐限制但未饱和：减少该单元上的工作量或提高其效率",
    "large_headroom": "离下限很远（<50%）且不受吞吐限制：瓶颈是延迟，优先查延迟成因（L3 SQTT）或换实现结构",
    "moderate_headroom": "离下限 50–80%：仍有实现层面的空间，按瓶颈类型逐项优化",
    "bound_violated": "实测快于下限：标定峰值偏低或模型假设不成立，需复核（审计信号，不是结论）",
}


def fusion_floor(calls: list[dict], groups: dict, M: "Machine") -> dict | None:
    """Per phase (op_spec group 'phase'), bytes of tensors crossing the phase boundary:
    inputs not produced inside the phase + outputs not consumed later inside the phase.
    If each phase were ONE ideal kernel keeping every intermediate on chip, this is its
    traffic -- a loose floor showing what the multi-kernel decomposition costs."""
    if not calls or "ptr" not in (calls[0].get("inputs") or [{}])[0]:
        return None
    phases: dict[str, list] = {}
    for c in calls:
        g = groups.get(c["launcher"])
        if g:
            phases.setdefault(g.get("phase", "all"), []).append(c)
    out = {}
    for ph, cs in phases.items():
        last_read: dict[int, int] = {}
        for i, c in enumerate(cs):
            for t in c["inputs"]:
                last_read[t["ptr"]] = i
        produced: set = set()
        ext_in: dict[int, int] = {}
        ext_out = 0
        for i, c in enumerate(cs):
            for t in c["inputs"]:
                if t["ptr"] not in produced:          # comes from outside the phase
                    ext_in[t["ptr"]] = t["bytes"]
            for t in c["outputs"]:
                produced.add(t["ptr"])
                if last_read.get(t["ptr"], -1) <= i:  # never read later inside the phase
                    ext_out += t["bytes"]
        b_in, b_out = sum(ext_in.values()), ext_out
        out[ph] = {"in_MB": b_in / 1e6, "out_MB": b_out / 1e6,
                   "T_us": (b_in + b_out) / M.bw * 1e6}
    return out


def analyze(machine: dict, pmc: dict, spec: dict, optrace: dict, shape: dict) -> dict:
    M, th = Machine(machine), thresholds()
    env = dict(shape)
    for k, v in (spec.get("vars") or {}).items():
        env[k] = _eval(v, env)
    groups = {g["launcher"]: g for g in spec["groups"]}
    entries = align(optrace["calls"], groups, pmc["kernels"])
    rows = []
    for e in entries:
        g = e["spec"] or {}
        serial = g.get("serial")
        kms = []
        for k in e["kernels"]:
            km = kernel_metrics(k, M, th)
            km["name"] = short_name(k["name"])
            km["classification"] = classify(km, bool(serial), th, M.cus)
            kms.append(km)
        t = sum(k["t_us"] for k in kms)
        T_impl = sum(max(k["T"]["par_mem_impl"], k["T"]["par_pipe"], k["T"]["mmac_impl"], k["T"]["launch"])
                     for k in kms)
        comp = {"launch": M.launch_us * len(kms)}
        if e["call"]:
            # the group's interface traffic could all be moved by its widest kernel
            bw_par = M.bw_for(max(k["ctas"] for k in kms))
            iface = e["call"]["in_bytes"] + e["call"]["out_bytes"]
            comp["mem_min"] = iface / M.bw * 1e6
            comp["par_mem_min"] = iface / bw_par * 1e6
        cp = {}
        if serial:
            steps = _eval(serial["steps"], env)
            blk = kms[0]["block"]
            cp["steps"] = steps
            cp["chain_impl_cycles"] = M.chain_cycles(serial["chain"], blk)
            cp["chain_min_cycles"] = M.chain_cycles(serial.get("chain_min", serial["chain"]), blk)
            cp["T_cp_impl"] = steps * cp["chain_impl_cycles"] / M.clk * 1e6
            cp["T_cp_min"] = steps * cp["chain_min_cycles"] / M.clk * 1e6
            comp["cp_min"] = cp["T_cp_min"]
            T_impl = max(T_impl, cp["T_cp_impl"])
        prior_terms = {k: v for k, v in comp.items() if k in ("par_mem_min", "cp_min", "launch")}
        if "par_mem_min" not in prior_terms:      # unattributed kernel: no interface data
            prior_terms["par_mem_impl"] = sum(k["T"]["par_mem_impl"] for k in kms)
        T_prior = max(prior_terms.values())
        binding = max(prior_terms, key=prior_terms.get)
        att_p, att_i = T_prior / t, min(1.0, T_impl / t) if t else 0.0
        dom = max(kms, key=lambda k: k["t_us"])["classification"]["class"]
        rows.append({
            "launcher": e["launcher"] or "(unattributed)",
            "kernels": kms,
            "t_us": t,
            "interface_MB": None if not e["call"] else {"in": e["call"]["in_bytes"] / 1e6, "out": e["call"]["out_bytes"] / 1e6},
            "components_us": comp, "critical_path": cp,
            "T_lower_prior_us": T_prior, "binding_prior": binding,
            "T_lower_impl_us": T_impl,
            "attainment_prior": att_p, "attainment_impl": att_i,
            "headroom_x": t / T_prior if T_prior else None,
            "recoverable_us": max(0.0, t - T_prior),
            "dominant_class": dom,
            "verdict": verdict(att_p, att_i, dom, th),
        })
    tot_t = sum(r["t_us"] for r in rows)
    tot_p = sum(r["T_lower_prior_us"] for r in rows)
    tot_i = sum(r["T_lower_impl_us"] for r in rows)
    ff = fusion_floor(optrace["calls"], groups, M)
    audit = [f"{r['launcher']}: measured {r['t_us']:.1f} us < prior bound {r['T_lower_prior_us']:.1f} us"
             for r in rows if r["verdict"] == "bound_violated"]
    return {
        "schema": "kra.waterlevel/1",
        "operator": spec.get("operator"), "shape": shape,
        "machine": machine.get("id"),
        "totals": {"t_us": tot_t, "T_lower_prior_us": tot_p, "T_lower_impl_us": tot_i,
                   "attainment_prior": tot_p / tot_t if tot_t else None,
                   "attainment_impl": tot_i / tot_t if tot_t else None,
                   "headroom_x": tot_t / tot_p if tot_p else None},
        "fusion_floor": ff,
        "critical_path_min_us": sum(r["critical_path"].get("T_cp_min", 0.0) for r in rows),
        "audit": audit,
        "priority": sorted(({"launcher": r["launcher"], "recoverable_us": r["recoverable_us"],
                             "share_of_total": r["recoverable_us"] / tot_t if tot_t else 0}
                            for r in rows), key=lambda x: -x["recoverable_us"]),
        "rows": rows,
        "notes": [
            "t_us: PMC run kernel time (EndNs-BeginNs, min of the read/write replays)",
            "prior bound = max(interface bytes at attainable BW for the grid, dependency-only critical path, launch)",
            "impl bound = same work as measured, at 100% of attainable bandwidth / VALU-pipe issue rate",
            "compute prior is not modelled per kernel; implementation MMAC work is in T.mmac_impl",
            "all HBM bytes use the single highest measured bandwidth (mixed traffic beats the write/copy microbenchmarks)",
            "fusion floor: per phase, only tensors crossing the phase boundary at peak bandwidth (loose; ignores on-chip capacity)",
        ],
    }
