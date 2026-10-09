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
