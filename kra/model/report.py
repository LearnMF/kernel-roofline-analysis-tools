"""Markdown rendering of a water-level analysis (kra.waterlevel/1)."""
from __future__ import annotations

from .bounds import VERDICT_TEXT


def _f(x, nd=1):
    return "—" if x is None else f"{x:,.{nd}f}"


def to_markdown(res: dict, title: str | None = None) -> str:
    tot = res["totals"]
    shape = ", ".join(f"{k}={v}" for k, v in res["shape"].items())
    L = [f"# {title or 'Water level (L2+L4)'}", "",
         f"operator: {res['operator']} | shape: {shape} | machine: `{res['machine']}`", "",
         "## Operator total", "",
         f"- measured kernel time: **{tot['t_us']:,.0f} µs**",
         f"- prior lower bound (interface bytes / dependency chain / launch): **{tot['T_lower_prior_us']:,.0f} µs** "
         f"→ attainment **{tot['attainment_prior']:.0%}**, headroom ≤ **{tot['headroom_x']:.2f}×**",
         f"- implementation lower bound (same measured work at 100% of attainable BW / pipe): "
         f"{tot['T_lower_impl_us']:,.0f} µs → attainment {tot['attainment_impl']:.0%}"]
    ff = res.get("fusion_floor")
    if ff:
        parts = "; ".join(f"{ph}: {v['in_MB']:,.0f} MB in + {v['out_MB']:,.0f} MB out → {v['T_us']:,.0f} µs"
                          for ph, v in ff.items())
        L.append(f"- fusion floor (each phase as ONE ideal kernel, boundary tensors only): {parts}; "
                 f"total {sum(v['T_us'] for v in ff.values()):,.0f} µs; dependency-only critical paths "
                 f"{res.get('critical_path_min_us', 0):,.0f} µs")
    if res.get("audit"):
        L += ["", "**Audit** (bound violated — check calibration / model):"] + [f"- {a}" for a in res["audit"]]
    c = res.get("compute")
    if c:
        L += ["", "## Compute angle (MFU / HFU)", "",
              f"- model FLOPs (algorithm, fwd + bwd = 3x fwd): **{c['model_GFLOP']:,.1f} GFLOP** → compute floor "
              f"{c['T_model_compute_us']:,.0f} µs at the attainable {c['peak_attainable_TFLOPS']:,.0f} TFLOPS",
              f"- **MFU {c['MFU_vs_theoretical']:.1%}** of the theoretical {c['peak_theoretical_TFLOPS']:,.0f} TFLOPS "
              f"({c['MFU_vs_attainable']:.1%} of attainable)",
              f"- HFU (MMAC FLOPs actually executed, incl. recompute / padding: {c['executed_mmac_GFLOP']:,.1f} GFLOP): "
              f"{c['HFU_vs_theoretical']:.1%}",
              "- reading: a low MFU with high HBM / VMEM-issue / pipe utilization means the operator is NOT "
              "compute-bound; MFU is then a reporting number, not an optimization target"]
    g, st = res.get("gap_decomposition"), res.get("strategy")
    if g and st:
        t = tot["t_us"]
        L += ["", "## Ceiling verdict: where the operator's time is (local vs global)", "",
              "| tier | µs | share | what removes it |", "|---|---:|---:|---|",
              f"| physical floor (boundary tensors / dependency chains, any decomposition) | "
              f"{g['physical_floor_us']:,.0f} | {g['physical_floor_us'] / t:.0%} | nothing (hardware) |",
              f"| decomposition cost (this launcher split vs the physical floor) | {g['decomposition_us']:,.0f} | "
              f"{g['decomposition_us'] / t:.0%} | fusion / fewer HBM round trips / more parallel grids |",
              f"| excess work (measured bytes+instructions vs the interface minimum) | {g['excess_work_us']:,.0f} | "
              f"{g['excess_work_us'] / t:.0%} | layout / redundancy removal inside launchers |",
              f"| execution inefficiency (stalls on the work done) | {g['execution_inefficiency_us']:,.0f} | "
              f"{g['execution_inefficiency_us'] / t:.0%} | local tuning (waits, scheduling, occupancy) |",
              "", f"**Largest tier: {st['largest'].replace('_us', '')} → {st['advice']}.**"]
        L += ["", "| launcher | binding angle | HBM | pipe | MMAC (HFU) | VMEM issue | LDS wait | chain | CUs busy | verdict | next direction |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
        for r in res["rows"]:
            a = r.get("ceiling_angles") or {}
            L.append(f"| {r['launcher']} | {r.get('binding_angle', '-')} | {a.get('hbm', 0):.0%} | {a.get('pipe', 0):.0%} | "
                     f"{a.get('mmac_hfu', 0):.1%} | {a.get('vmem_issue', 0):.0%} | {a.get('lds_wait', 0):.0%} | "
                     f"{('%.0f%%' % (100 * a['chain'])) if 'chain' in a else '—'} | {a.get('parallelism', 0):.0%} | "
                     f"{r['verdict']} | {r.get('next_direction', '')} |")
    L += ["",
         "## Per launcher", "",
         "| launcher | kernels | t µs | class | pipe / HBM SOL | T_mem_min | T_par_mem_min | T_cp_min / impl | "
         "prior bound (binding) | att. prior | impl bound | att. impl | verdict |",
         "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in res["rows"]:
        c = r["components_us"]
        cp = r["critical_path"]
        cls = "<br>".join(
            k["classification"]["class"] + ("/" + "+".join(s.split(".", 1)[1] for s in k["classification"]["subtypes"])
                                            if k["classification"]["subtypes"] else "")
            for k in r["kernels"])
        sol = "<br>".join(f"{k['sol_pipe']:.0%} / {k['sol_mem']:.0%}" for k in r["kernels"])
        names = "<br>".join(k["name"] for k in r["kernels"])
        cps = f"{_f(cp.get('T_cp_min'))} / {_f(cp.get('T_cp_impl'))}" if cp else "—"
        L.append(f"| {r['launcher']} | {names} | {_f(r['t_us'])} | {cls} | {sol} | {_f(c.get('mem_min'))} | "
                 f"{_f(c.get('par_mem_min'))} | {cps} | {_f(r['T_lower_prior_us'])} ({r['binding_prior']}) | "
                 f"{r['attainment_prior']:.0%} | {_f(r['T_lower_impl_us'])} | {r['attainment_impl']:.0%} | "
                 f"{r['verdict']} |")
    L += ["", "## Where the time above the bound is (priority by recoverable µs)", "",
          "| launcher | recoverable µs (t − prior bound) | share of operator time |", "|---|---:|---:|"]
    for p in res["priority"]:
        if p["recoverable_us"] > 0:
            L.append(f"| {p['launcher']} | {p['recoverable_us']:,.0f} | {p['share_of_total']:.1%} |")
    L += ["", "## Kernel details", "",
          "| kernel | CTAs | t µs | HBM GB/s | rd/wr MB | pipe SOL (active CUs) | HBM SOL (active CUs) | "
          "MMAC slot share | LDS conflict | VGPR | causes / notes |",
          "|---|---:|---:|---:|---|---|---|---:|---:|---:|---|"]
    for r in res["rows"]:
        for k in r["kernels"]:
            cl = k["classification"]
            notes = "; ".join([f"{c['id']}={c['value']}" for c in cl["causes"]] + cl["reasons"][1:])
            L.append(f"| {k['name']} | {k['ctas']} | {_f(k['t_us'])} | {_f(k['hbm_GBps'], 0)} | "
                     f"{_f(k['hbm_rd_MB'])}/{_f(k['hbm_wr_MB'])} | {k['sol_pipe']:.0%} ({k['sol_pipe_active']:.0%}) | "
                     f"{k['sol_mem']:.0%} ({k['sol_mem_active']:.0%}) | {k['mmac_slot_share']:.0%} | "
                     f"{k['lds_conflict_ratio']:.1%} | {k['vgpr']} | {notes} |")
    L += ["", "## Verdict legend", ""] + [f"- `{k}`: {v}" for k, v in VERDICT_TEXT.items()]
    L += ["", "## Notes", ""] + [f"- {n}" for n in res["notes"]]
    return "\n".join(L) + "\n"
