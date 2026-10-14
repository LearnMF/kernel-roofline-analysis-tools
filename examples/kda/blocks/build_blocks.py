"""Parse the raw building-block measurements (results/kda-blocks-20261014/raw) into blocks.json.
    python3 build_blocks.py"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "..", "results", "kda-blocks-20261014")
RAW = os.path.join(RES, "raw")
sys.path.insert(0, HERE)


def kv(line):
    return dict(x.split("=", 1) for x in line.split() if "=" in x)


def num(v):
    try:
        return float(v)
    except ValueError:
        return v


def lines(name):
    p = os.path.join(RAW, name)
    return open(p).read().splitlines() if os.path.exists(p) else []


def main():
    import isa_sim as sim
    B = {"meta": {"machine": "gfx936-bw1000", "date": "2026-10-14", "clock_ghz": 1.5, "gpu": "bw52",
                  "fingerprint_ref": "90a91fdd5fdd",
                  "clock_note": "s_memtime ticks at the 1.50 GHz shader clock under load; s_memrealtime is 25 MHz"},
         "vmem_load": [], "vmem_store": [], "lds": [], "lds_exchange": [], "barrier": [], "mmac": [], "valu": []}
    for l in lines("vmem_bench.txt"):
        if not l.startswith("VMEM"):
            continue
        d = kv(l)
        base = dict(source=d["src"], bytes_per_lane=int(d["bpl"]), rows=int(d["rows"]), row_bytes=int(d["row_bytes"]),
                    dup=int(d["dup"]), pitch=int(d["pitch"]), mode=d["mode"])
        B["vmem_load"].append(dict(base, ta_cycles_per_inst=float(d["ld_cyc"]), gbs=float(d["ld_GBs"]),
                                   latency_cycles=float(d["lat_cyc"]) if float(d["lat_cyc"]) > 0 else None))
        B["vmem_store"].append(dict(base, ta_cycles_per_inst=float(d["st_cyc"]), gbs=float(d["st_GBs"])))
    for l in lines("lds_bench.txt"):
        d = kv(l)
        if l.startswith("LDSOP"):
            B["lds"].append(dict(op=d["op"], bytes=int(d["bytes"]), stride=int(d["stride"]), ways=int(d["ways"]),
                                 cycles_per_inst=float(d["cyc_per_inst_cu"]), bytes_per_cycle_per_cu=float(d["bytes_per_cyc_cu"])))
        elif l.startswith("XCHG"):
            B["lds_exchange"].append(dict(waves=int(d["W"]), ctas_per_cu=int(d["ctas_per_cu"]), bytes=int(d["bytes_per_wave"]),
                                          pattern="conflict_free_b128", us=float(d["us_per_step"]), cycles=float(d["cyc_per_step"])))
        elif l.startswith("BAR"):
            B["barrier"].append(dict(waves=int(d["W"]), skew_valu=int(d["skew_valu"]), cycles=float(d["cyc_per_barrier"])))
        elif l.startswith("LDSLAT"):
            B["lds_latency_dependent_read_cycles"] = float(re.search(r"([\d.]+) cycles", l).group(1))
        elif l.startswith("LDSCOMPL"):
            B.setdefault("lds_completion_cycles", {})[d["op"]] = float(d["issue_to_lgkmcnt0_cycles"])
    for l in lines("mmac_valu_bench.txt"):
        d = kv(l)
        if l.startswith("MMAC"):
            B["mmac"].append(dict(form=d["form"], indep_chains=int(d["indep"]), valu_per_mmac=int(d["valu_per_mmac"]),
                                  waves_per_simd=int(d["waves_per_simd"]), cycles_per_mmac_wave=float(d["cyc_per_mmac_wave"]),
                                  cycles_per_mmac_simd=float(d["cyc_per_mmac_simd"]),
                                  note="loop of 1 MMAC per chain per iteration (loop-boundary hazards inflate the I=1 case)"))
        elif l.startswith("VALU"):
            B["valu"].append(dict(op=d["op"], context="4 waves / SIMD, inline asm", latency_cycles=float(d["lat_cyc"]),
                                  cycles_per_inst=float(d["thr_cyc_per_inst_simd"])))
    B["mmac_dependency"] = {}
    for l in lines("mmac_dep_bench.txt"):
        m = re.match(r"MMACDEP (\S+) cycles_per_dependent_mmac=([\d.]+)", l)
        if m:
            B["mmac_dependency"][m.group(1)] = float(m.group(2))
    B["vmem_completion_cycles"] = {}
    for l in lines("vmem_compl_bench.txt"):
        m = re.match(r"VMEMCOMPL (\S+) ws=(\d+)MB issue_to_vmcnt0_cycles=(\d+)", l)
        if m:
            B["vmem_completion_cycles"][f"{m.group(1)}_ws{m.group(2)}MB"] = float(m.group(3))
    B["issue_queue"] = {
        "vmem_per_wave_queue": sim.Q_WAVE,
        "note": "issue_burst_bench (SQTT): a single wave issues ~16-20 VMEM / LDS instructions at 4 cycles each, then at "
                "the unit's acceptance rate; no hard in-flight limit up to 32 outstanding loads",
        "vmem_dwordx4_acceptance_cycles": {"1_wave": 16.0, "2_waves_per_cu": 30.0, "32_waves_per_cu_saturated": 22.0},
        "lds_single_wave_cycles": sim.LDS_WAVE,
        "raw": "raw/issue_burst.txt"}
    ov = {"launch_gap_us": {}, "tail_rounds": []}
    for l in lines("overhead_bench.txt"):
        d = kv(l)
        if l.startswith("LAUNCH"):
            ov["launch_gap_us"][d["n"]] = float(d["us_per_launch"])
        elif l.startswith("MIN"):
            ov["min_kernel_us"] = {"empty": float(d["empty_us"]), "one_load_store_80ctas": float(d["one_load_store_80ctas_us"])}
        elif l.startswith("TAIL"):
            ov["tail_rounds"].append(dict(ctas=int(d["ctas"]), us=float(d["us"]), rounds=int(d["rounds"])))
        elif l.startswith("DISPATCH"):
            ov.setdefault("cta_dispatch_ns_per_cta_per_cu", {})[d["ctas"]] = float(d["ns_per_cta_per_cu"])
    ov["tail_wave_rule"] = "time = rounds x per-round time, rounds = ceil(ctas / (80 x ctas_per_cu)); measured steps exact to <0.3%"
    B["kernel_overhead"] = ov
    B["recurrence_step"] = {
        "model": "ISA-level in-order issue simulation of one CTA (isa_sim.py) on the production loop body extracted from "
                 "SQTT (sq_loop.py); costs: COST / LDS_OCC / TA_OCC / MMAC_FWD / LDS_WAVE / Q_WAVE in isa_sim.py, all from "
                 "this library; per-instance memory input hbm_frac = unique HBM bytes / loaded bytes",
        "params": {"COST": sim.COST, "MMAC_FWD": sim.MMAC_FWD, "LDS_OCC": sim.LDS_OCC, "LDS_LAT": sim.LDS_LAT,
                   "TA_OCC": sim.TA_OCC, "BAR": sim.BAR, "ISSUE": sim.ISSUE, "Q_WAVE": sim.Q_WAVE,
                   "LDS_WAVE": sim.LDS_WAVE, "vmem_lat_load": 900.0, "vmem_lat_store": 430.0},
        "replicas": json.load(open(os.path.join(RES, "validate_recurrence.json")))
        if os.path.exists(os.path.join(RES, "validate_recurrence.json")) else [],
    }
    B["references"] = {"occupancy_curves": "results/kda-decomp-calib-20261013/calib.json#p1",
                       "register_spill": "results/kda-decomp-calib-20261013/calib.json#p2"}
    json.dump(B, open(os.path.join(RES, "blocks.json"), "w"), indent=1)
    print("blocks.json:", {k: len(v) if isinstance(v, (list, dict)) else v for k, v in B.items() if k != "meta"})


if __name__ == "__main__":
    main()
