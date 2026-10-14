"""Query interface of the gfx936 building-block library (results/kda-blocks-20261014/blocks.json).
Only measured points are returned; queries outside the measured grid raise OutOfRange unless extrapolate=True
(then the result carries 'extrapolated': True).

    vmem_cost(bytes_per_lane, rows, row_bytes, source, store=False, pitch=None, mode="scatter")
    lds_cost(op, stride=None)                    lds_exchange_us(bytes, waves, ctas_per_cu)
    barrier_cycles(waves, skew_cycles=0)         mmac_cycles(form, indep_chains, waves_per_simd)
    mmac_dependency()                            valu_cycles(op)
    step_us(loop_body_s, waves, hbm_frac=0, active_cus=80, ctas_per_cu=1)   kernel_overhead()
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_B = json.load(open(os.path.join(_HERE, "..", "..", "results", "kda-blocks-20261014", "blocks.json")))


class OutOfRange(KeyError):
    pass


def _nearest(cands, key, val):
    return min(cands, key=lambda r: abs(r[key] - val))


def vmem_cost(bytes_per_lane, rows, row_bytes, source, store=False, pitch=None, mode="scatter", extrapolate=False):
    """CU cycles of TA / memory occupancy per wave instruction (+ latency when measured).  source: 'l2' | 'hbm'."""
    tab = _B["vmem_store" if store else "vmem_load"]
    c = [r for r in tab if r["source"] == source and r["bytes_per_lane"] == bytes_per_lane and r["rows"] == rows
         and r["mode"] == mode and r["row_bytes"] == row_bytes]
    if not c:
        if not extrapolate:
            raise OutOfRange(f"vmem {bytes_per_lane}B/lane {rows}x{row_bytes}B {source} {mode} not measured")
        c = [r for r in tab if r["source"] == source and r["mode"] == mode]
        r = min(c, key=lambda r: abs(r["rows"] - rows) * 100 + abs(r["bytes_per_lane"] - bytes_per_lane))
        return dict(ta_cycles=r["ta_cycles_per_inst"], latency_cycles=r.get("latency_cycles"), extrapolated=True)
    r = _nearest(c, "pitch", pitch) if pitch else min(c, key=lambda r: r["pitch"])
    return dict(ta_cycles=r["ta_cycles_per_inst"], latency_cycles=r.get("latency_cycles"), pitch=r["pitch"],
                extrapolated=False)


def lds_cost(op, stride=None):
    c = [r for r in _B["lds"] if r["op"].startswith(op)]
    if not c:
        raise OutOfRange(f"lds op {op} not measured")
    r = _nearest(c, "stride", stride) if stride else min(c, key=lambda r: r["cycles_per_inst"])
    return dict(cycles=r["cycles_per_inst"], latency=_B.get("lds_latency_dependent_read_cycles"), stride=r["stride"])


def lds_exchange_us(nbytes, waves, ctas_per_cu=1, extrapolate=False):
    c = sorted([r for r in _B["lds_exchange"] if r["waves"] == waves and r["ctas_per_cu"] == ctas_per_cu],
               key=lambda r: r["bytes"])
    if not c or not (c[0]["bytes"] <= nbytes <= c[-1]["bytes"]) and not extrapolate:
        raise OutOfRange(f"lds exchange {nbytes} B x {waves} waves x {ctas_per_cu} CTAs/CU outside the grid")
    for a, b in zip(c, c[1:]):
        if a["bytes"] <= nbytes <= b["bytes"]:
            f = (nbytes - a["bytes"]) / (b["bytes"] - a["bytes"])
            return a["us"] + f * (b["us"] - a["us"])
    r = _nearest(c, "bytes", nbytes)
    return r["us"] * nbytes / r["bytes"]


def barrier_cycles(waves, skew_cycles=0):
    c = [r for r in _B["barrier"] if r["waves"] == waves and r["skew_valu"] == 0]
    if not c:
        raise OutOfRange(f"barrier with {waves} waves not measured")
    return c[0]["cycles"] + skew_cycles


def mmac_cycles(form, indep_chains, waves_per_simd, valu_per_mmac=0):
    c = [r for r in _B["mmac"] if r["form"] == form and r["indep_chains"] == indep_chains
         and r["waves_per_simd"] == waves_per_simd and r["valu_per_mmac"] == valu_per_mmac]
    if not c:
        raise OutOfRange(f"mmac {form} I={indep_chains} wps={waves_per_simd} V={valu_per_mmac} not measured")
    return dict(per_wave=c[0]["cycles_per_mmac_wave"], per_simd=c[0]["cycles_per_mmac_simd"])


def mmac_dependency():
    """{'srcC_chain_unrolled': cycles, 'result->VALU->srcA': cycles} (straight-line code)"""
    return dict(_B["mmac_dependency"])


def valu_cycles(op):
    c = [r for r in _B["valu"] if r["op"] == op]
    if not c:
        raise OutOfRange(f"valu op {op} not measured")
    return dict(cycles=c[0]["cycles_per_inst"], latency=c[0]["latency_cycles"])


def step_us(loop_body_s, waves, hbm_frac=0.0, active_cus=80, ctas_per_cu=1):
    """recurrence step time (us) of a loop body (.s with a 'Loop Header' block) by the ISA simulator"""
    sys.path.insert(0, os.path.join(_HERE, "blocks"))
    import isa_sim  # noqa: PLC0415
    cyc, _ = isa_sim.simulate(isa_sim.parse_loop(loop_body_s), waves, 16, hbm_frac=hbm_frac, active_cus=active_cus,
                              ctas_per_cu=ctas_per_cu)
    return cyc / 1500.0


def kernel_overhead():
    return dict(_B["kernel_overhead"])
