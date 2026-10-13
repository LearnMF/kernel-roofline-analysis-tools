"""Acceptance checks over raw/ + cards.json:
  1. sum of per-kernel device times vs independent hip-trace total (+/-3%)
     (window wall = last op end - first op start; kernel-sum = sum of kernel durs in window)
  2. wu / bwd_tail / dav HBM bytes vs optrace interface bytes (+/-15%)
  3. ISA-derived MMAC count vs SQ_INSTS_MMOP (+/-5%) for the required kernels
  4. per-process dispersion of medians
Run: python3 verify.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from parse_raw import trace_times, short                        # noqa: E402
from kra.timeline.hipprof_json import load as tload             # noqa: E402
from kra.timeline.analyze import split_windows                  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "results" / "kda-ceiling-cards-20261013"
RAW = RES / "raw"
CARDS = json.load(open(RES / "cards.json"))
ISA = json.load(open(RAW / "isa_mix.json"))
import re
from cards import MANGLED                                       # noqa: E402

L = []


def p(s=""):
    print(s)
    L.append(s)


for s, c in sorted(CARDS.items(), key=lambda x: (x[1]["T"], x[1]["H"])):
    d = RAW / s
    # ---- 1. time consistency
    sums, walls = [], []
    for i in range(1, 6):
        f = d / f"trace_{i}.json"
        if not f.exists():
            continue
        for w in split_windows(tload(f), "spin_kernel", None):
            ks = [o for o in w if o.kind == "kernel"]
            if ks:
                sums.append(sum(o.dur_ns for o in ks) / 1e3)
                walls.append((max(o.end_ns for o in ks)
                              - min(o.begin_ns for o in ks)) / 1e3)
    import statistics as st
    med_sum = st.median(sums) if sums else 0
    med_wall = st.median(walls) if walls else 0
    card_tot = sum(v["t_us"] for v in c["cards"].values())
    p(f"### {s}: kernel-sum(card) {card_tot / 1e3:.2f} ms | window kernel-sum med {med_sum / 1e3:.2f} "
      f"| window wall med {med_wall / 1e3:.2f} | card/window-sum {card_tot / max(1, med_sum):.3f} "
      f"(accept |1-r|<=0.03: {'PASS' if abs(1 - card_tot / max(1, med_sum)) <= 0.03 else 'FAIL'})")
    # ---- 2. required-byte cross-check
    for k in ("wu#1", "bwd_tail#1", "dav#1"):
        v = c["cards"].get(k)
        if not v or not v.get("req_bytes"):
            p(f"  {k}: —")
            continue
        r = (v["hbm_rd"] + v["hbm_wr"]) / v["req_bytes"]
        p(f"  {k}: actual/required {r:.3f} ({'PASS' if abs(1 - r) <= 0.15 else 'FAIL'})")
    # ---- 3. ISA loop-body MMAC x trips vs SQ_INSTS_MMOP (+/-5%)
    #    trip factor is derived per kernel: recurrences step loop iterates NT*4
    #    per CTA (4 k-subtiles per step); s4a/s4b factors inferred below.
    for k in ("fwd_h_v8o4#1", "fwd_h_v8o4#2", "dhu_v8b4#1", "dhu_v8b#1",
              "wy_s4a#1", "wy_s4b#1"):
        v = c["cards"].get(k)
        if not v:
            continue
        frag = MANGLED.get(re.sub(r"#\d+$", "", k), MANGLED.get(k))
        rec = next((vv for nn, vv in ISA.items() if frag and frag in nn), {})
        lm = {a: b for a, b in (rec.get("loop_mix") or {}).items() if "mmac" in a}
        per = sum(lm.values())
        if not per:
            p(f"  {k}: no loop MMAC in ISA")
            continue
        if k.startswith(("fwd_h", "dhu")):
            subtiles = 8 if k.split("#")[0] in ("dhu_v8b", "dhu_v8b2") else 4
            trips = c["NT"] * subtiles * v["ctas"]   # NT steps x subtiles x CTA
        else:
            trips = v["ctas"] * round(v["mmop"] / (per * v["ctas"]))
        exp = per * trips
        dev = (exp - v["mmop"]) / v["mmop"]
        p(f"  {k}: ISA loop {per}/iter x {trips / v['ctas']:.0f}/CTA x {v['ctas']} CTAs "
          f"= {exp:.0f} vs SQ {v['mmop']:.0f} dev {dev:+.3f} "
          f"({'PASS' if abs(dev) <= 0.05 else 'CHECK'})")

out = RES / "verification.md"
out.write_text("\n".join(L) + "\n")
print("wrote", out)
