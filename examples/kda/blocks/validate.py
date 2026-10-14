"""Cross-validation of the gfx936 building-block library (docs/tasks/kda_blocks_task.md section 4).
  1. recurrence step: ISA simulation of the 4 production recurrence instances (loop bodies from SQTT) vs the measured
     per-step time of the ceiling cards (no fitting to these kernels; per-instance memory input hbm_frac = HBM read
     bytes of the card / bytes loaded by the loop)
  2. VMEM: bwd_tail TA cycles per wave from its access inventory (kda_layout.py) x library costs vs the card's TA busy
  3. sanity: library saturation values vs the machine ceilings
    python3 validate.py"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
RES = os.path.join(ROOT, "results", "kda-blocks-20261014")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import isa_sim  # noqa: E402

CARDS = json.load(open(os.path.join(ROOT, "results", "kda-ceiling-cards-20261013", "cards.json")))
BODIES = os.path.join(RES, "raw", "sqtt_bodies")
WIDTH = {"dword": 4, "dwordx2": 8, "dwordx4": 16, "ushort": 2, "short": 2}


def loaded_bytes_per_wave(body):
    b = 0
    for opc, ops, cls in body:
        if cls == "vld":
            b += 64 * WIDTH.get(opc.split("_", 2)[-1], 8)
    return b


def recurrences():
    inst = [  # name, body, waves, shape, card, rounds
        ("fwd MODE 1 (fwd_h_o)", "sq_body_mode1.s", 4, "16384_12", "fwd_h_v8o4#1", 1),
        ("recompute MODE 4 (fwd_h_bn)", "sq_body_mode4.s", 4, "16384_12", "fwd_h_v8o4#2", 1),
        ("dhu 8-wave", "sq_body_dhu8.s", 8, "16384_12", "dhu_v8b#1", 1),
        ("dhu 4-wave (P-24 throttled, 1 CTA/CU)", "sq_body_dhu4.s", 4, "16384_48", "dhu_v8b4#1", 3),
    ]
    rows = []
    for name, f, W, sh, card, rounds in inst:
        c = CARDS[sh]["cards"][card]
        NT = CARDS[sh]["NT"]
        body = isa_sim.parse_loop(os.path.join(BODIES, f))
        loaded = loaded_bytes_per_wave(body) * W * NT * c["ctas"]
        hf = min(1.0, c["hbm_rd"] / loaded)
        active = min(80, c["ctas"])
        cyc, _ = isa_sim.simulate(body, W, 16, hbm_frac=hf, active_cus=active)
        meas = c["t_us"] / (NT * rounds)
        rows.append(dict(name=name, body=f, waves=W, shape=sh, hbm_frac=round(hf, 3), active_cus=active,
                         predicted_us=round(cyc / 1500, 4), measured_us=round(meas, 4),
                         error=round(cyc / 1500 / meas - 1, 4)))
    json.dump(rows, open(os.path.join(RES, "validate_recurrence.json"), "w"), indent=1)
    print("## 1. recurrence step (ISA simulation vs measured)\n")
    print("| instance | hbm_frac | predicted us/step | measured us/step | error |")
    print("|---|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['name']} | {r['hbm_frac']} | {r['predicted_us']:.3f} | {r['measured_us']:.3f} | {100 * r['error']:+.0f}% |")
    return rows


def tail_ta():
    import blocks_model as bm
    import kda_layout as kl
    print("\n## 2. bwd_tail TA cycles per wave (inventory x library costs, data from HBM) vs the card\n")
    print("| shape | predicted | measured (TA busy / wave) | error | inventory VMEM / wave vs PMC |")
    print("|---|---:|---:|---:|---|")
    out = []
    for sh in ("16384_12", "16384_48"):
        H = int(sh.split("_")[1])
        c = CARDS[sh]["cards"]["bwd_tail#1"]
        waves = c["ctas"] * c["block"] // 64
        tot = n = 0.0
        for a in kl.INVENTORY:
            if a.kernel != "bwd_tail":
                continue
            p = kl.pattern(a.role, a.layout, a.store)
            if a.role == "rowvec16":
                r = [x for x in bm._B["vmem_store" if a.store else "vmem_load"] if x["mode"] == "rowvec16" and x["source"] == "hbm"]
                cost = min(r, key=lambda x: abs(x["pitch"] - H * 4))["ta_cycles_per_inst"]
            else:
                dt = 4 if a.role in ("row2f",) else 2
                mode = "stream" if p.rows == 1 else "scatter"
                cost = bm.vmem_cost(p.bpl, p.rows, p.row_bytes, "hbm", a.store, pitch=H * 128 * dt, mode=mode,
                                    extrapolate=True)["ta_cycles"]
            tot += cost * a.n
            n += a.n
        meas = c["ta_busy"] / waves
        out.append((sh, tot, meas))
        print(f"| {sh} | {tot:.0f} | {meas:.0f} | {100 * (tot / meas - 1):+.0f}% | {n:.0f} vs {c['vmem'] / waves:.1f} |")
    return out


def sanity():
    B = json.load(open(os.path.join(RES, "blocks.json")))
    clk, cus = 1.5e9, 80
    ld = [r for r in B["vmem_load"] if r["source"] == "hbm" and r["bytes_per_lane"] == 16 and r["rows"] == 1 and r["mode"] == "scatter"][0]
    st = [r for r in B["vmem_store"] if r["source"] == "hbm" and r["bytes_per_lane"] == 16 and r["rows"] == 1 and r["mode"] == "scatter"][0]
    l2 = [r for r in B["vmem_load"] if r["source"] == "l2" and r["bytes_per_lane"] == 16 and r["rows"] == 1 and r["mode"] == "scatter"][0]
    lds = max(r["bytes_per_cycle_per_cu"] for r in B["lds"])
    mm = min(r["cycles_per_mmac_simd"] for r in B["mmac"] if r["form"] == "bf16_16x16x16" and r["valu_per_mmac"] == 0)
    rows = [("HBM read (1 KB rows)", 1024 / ld["ta_cycles_per_inst"] * cus * clk / 1e12, 1.3392, "TB/s"),
            ("HBM write (1 KB rows)", 1024 / st["ta_cycles_per_inst"] * cus * clk / 1e12, 1.1005, "TB/s"),
            ("L2 read (1 KB rows)", 1024 / l2["ta_cycles_per_inst"] * cus * clk / 1e12, 6.1, "TB/s"),
            ("LDS (best op)", lds * cus * clk / 1e12, 15.2, "TB/s"),
            ("bf16 MMAC", 4 * 16 * 16 * 16 * 2 / mm * cus * clk / 1e12, 475.96, "TFLOPS")]
    print("\n## 3. sanity vs machine ceilings (machines/gfx936-bw1000.json)\n")
    print("| quantity | library | ceiling | ratio |")
    print("|---|---:|---:|---:|")
    for n, v, c, u in rows:
        print(f"| {n} | {v:.3f} {u} | {c} {u} | {v / c:.2f} |")
    return rows


if __name__ == "__main__":
    recurrences()
    tail_ta()
    sanity()
