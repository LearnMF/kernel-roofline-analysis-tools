"""Phase 2 step 3 acceptance: re-derive the production kernels from the building-block library only.
Inputs per kernel: its structure (production instruction streams from SQTT: one CTA's per-wave traces for the
chunk-parallel kernels, one steady-state step for the recurrences), its launch geometry and residency, and its
memory behaviour (hbm_frac = HBM read bytes / loaded bytes).  No measured kernel time enters a prediction.
    python3 derive11.py [--shapes 16384_12 16384_48]"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, HERE)
import isa_sim  # noqa: E402
import trace_sim  # noqa: E402

RES = os.path.join(ROOT, "results", "kda-blocks-20261014")
TRACES = os.path.join(RES, "raw", "cta_traces")
BODIES = os.path.join(RES, "raw", "sqtt_bodies")
CARDS = json.load(open(os.path.join(ROOT, "results", "kda-ceiling-cards-20261013", "cards.json")))

CHUNK = [("fwd_prep_a#1", "fwd_prep_a"), ("fwd_prep_b#1", "fwd_prep_b"), ("l2n_apply#1", "l2n_apply"),
         ("gate#1", "gate_kernel"), ("wu#1", "g2_wu_kernel"), ("dav#1", "g2_dav_kernel"), ("wy_s4a#1", "wy_s4a"),
         ("wy_s4b#1", "wy_s4b"), ("bwd_tail#1", "bwd_tail_kernel")]
REC = {"16384_12": [("fwd_h_v8o4#1", "sq_body_mode1.s", 4), ("fwd_h_v8o4#2", "sq_body_mode4.s", 4),
                    ("dhu_v8b#1", "sq_body_dhu8.s", 8)],
       "16384_48": [("dhu_v8b4#1", "sq_body_dhu4.s", 4)]}


def residency(c):
    """co-resident CTAs per CU from the resources: VGPR budget 768 per SIMD lane (R38: 3 waves at 200-256 VGPRs),
    at most 10 waves per SIMD, 64 KB LDS per CU (the card's 'resident' field is capped at 3)."""
    waves_cta = max(1, c["block"] // 64)
    vg = -(-c["vgpr"] // 8) * 8
    per_simd = min(10, 768 // max(vg, 8))
    by_vgpr = per_simd * 4 // waves_cta
    by_lds = 65536 // c["lds_c"] if c["lds_c"] else 64
    return max(1, min(by_vgpr, by_lds, 16))


def chunk(sh, card, name):
    c = CARDS[sh]["cards"].get(card)
    path = os.path.join(TRACES, name + ".json")
    if c is None or not os.path.exists(path):
        return None
    tr = json.load(open(path))
    progs = [trace_sim.decode_prog(w) for w in tr["waves"]]
    hf = min(1.0, c["hbm_rd"] / max(1.0, trace_sim.loaded_bytes(progs) * c["ctas"]))
    n = math.ceil(c["ctas"] / 80)
    res = residency(c)
    cyc = trace_sim.simulate(progs, res, n, hf, min(80, c["ctas"]))   # per-CU HBM queue (better validated than simulate_mf)
    return cyc / 1500.0, c["t_us"], hf, res


def rec(sh, card, body, W):
    c = CARDS[sh]["cards"].get(card)
    if c is None:
        return None
    NT = CARDS[sh]["NT"]
    b = isa_sim.parse_loop(os.path.join(BODIES, body))
    loaded = sum(64 * trace_sim.WIDTH.get(o.split("_", 2)[-1], 8) for o, _, cl in b if cl == "vld") * W * NT * c["ctas"]
    hf = min(1.0, c["hbm_rd"] / loaded)
    active = min(80, c["ctas"])
    res = c["resident"]
    per_cu = math.ceil(c["ctas"] / 80)
    rounds = math.ceil(per_cu / res)
    co = min(res, per_cu)              # CTAs sharing a CU at a time
    cyc, _ = isa_sim.simulate(b, W, 16, hbm_frac=hf, active_cus=active, ctas_per_cu=co)
    # co-resident CTAs on a CU: the simulator models one CTA with a 1/co bandwidth share; the SIMDs and the LDS / TA
    # are shared too, so the per-CTA step time is at least co x the single-CTA resource demand -- use max of the two
    t = cyc / 1500.0 * NT * rounds
    return t, c["t_us"], hf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=["16384_12", "16384_48"])
    a = ap.parse_args()
    out = []
    for sh in a.shapes:
        print(f"\n### {sh}\n")
        print("| kernel | kind | predicted us | measured us | error | hbm_frac |")
        print("|---|---|---:|---:|---:|---:|")
        for card, name in CHUNK:
            r = chunk(sh, card, name)
            if r:
                p, m, hf, res = r
                out.append(dict(shape=sh, kernel=card, kind="chunk", predicted=p, measured=m, error=p / m - 1, res=res))
                print(f"| {card} | chunk (trace, res {res}) | {p:.0f} | {m:.0f} | {100 * (p / m - 1):+.0f}% | {hf:.2f} |")
        for card, body, W in REC.get(sh, []):
            r = rec(sh, card, body, W)
            if r:
                p, m, hf = r
                out.append(dict(shape=sh, kernel=card, kind="recurrence", predicted=p, measured=m, error=p / m - 1))
                print(f"| {card} | recurrence (step) | {p:.0f} | {m:.0f} | {100 * (p / m - 1):+.0f}% | {hf:.2f} |")
        tp = sum(r["predicted"] for r in out if r["shape"] == sh)
        tm = sum(r["measured"] for r in out if r["shape"] == sh)
        print(f"| **sum of the kernels above** | | {tp:.0f} | {tm:.0f} | {100 * (tp / tm - 1):+.0f}% | |")
    json.dump(out, open(os.path.join(RES, "derive11.json"), "w"), indent=1)
    errs = [abs(r["error"]) for r in out]
    print(f"\n{len(out)} kernel-shape points: {sum(e <= 0.2 for e in errs)} within +-20%, {sum(e <= 0.3 for e in errs)} within +-30%")


if __name__ == "__main__":
    main()
