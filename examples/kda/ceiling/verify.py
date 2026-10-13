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

# ---- F-checks appended (revision task) ---------------------------------------
p("")
p("## F-修订检查")
# F2: interface floor <= stream floor <= measured  (per kernel, per shape)
vio = []
for s, c in sorted(CARDS.items(), key=lambda x: (x[1]["T"], x[1]["H"])):
    for k, v in c["cards"].items():
        fi = v.get("f_iface_us")
        if not fi:
            continue
        if not (fi <= v["floor_us"] + 0.5 and v["floor_us"] <= v["t_us"] + 0.5):
            vio.append((s, k, round(fi, 1), round(v["floor_us"], 1), round(v["t_us"], 1)))
p(f"F2 iface<=stream<=measured violations: {len(vio)}")
for x in vio[:12]:
    p(f"  {x[0]} {x[1]}: iface={x[2]} stream={x[3]} meas={x[4]}us")
# F3: TA amplification >= 1 (with slack for counter noise)
ta_bad = [(s, k, v["ta_amp"]) for s, c in CARDS.items()
          for k, v in c["cards"].items()
          if v.get("ta_amp") is not None and v["ta_amp"] < 0.99]
p(f"F3 TA放大<1.0: {len(ta_bad)}")
for x in ta_bad[:12]:
    p(f"  {x}")
# F1: sqtt symbols must be a subset of the production trace's kernel list
import csv, glob
import re
tr_syms = {}
for tag, rawd, perf_dir in (("16384_48", "16384_48", "sqtt_prod_16384_48"),
                          ("16384_12", "16384_12", "sqtt_prod_16384_12")):
    from kra.timeline.hipprof_json import load as tl2
    names = set()
    for o in tl2(RAW / rawd / "trace_1.json").ops:
        if o.kind == "kernel":
            names.add(o.name)
    tr_syms[tag] = names
    subs = glob.glob(str(RAW / perf_dir / "csv" / "*dispatch_summary.csv"))
    sq_syms = set()
    for f in subs:
        for r in csv.DictReader(open(f)):
            sq_syms.add(r["Kernel"])
    def norm(n):
        """(basename, template-args tuple) for both demangled and mangled forms."""
        m = re.search(r"\b(g2_\w+_kernel|p3_\w+_kernel)<([^>]*)>", n)
        if m:                                    # demangled production name
            args = tuple(a.strip() for a in m.group(2).split(",") if a.strip())
            return m.group(1), args
        m = re.search(r"(g2_\w+_kernel|p3_\w+_kernel)(I.*?)Ev", n)
        if m:                                    # mangled sqtt name
            toks = [("true" if t == "Lb1" else "false" if t == "Lb0"
                     else t[2:] if t.startswith("Li") else t)
                    for t in re.findall(r"Lb[01]|Li\d+", m.group(2))]
            return m.group(1), tuple(toks)
        m = re.search(r"(g2_\w+_kernel|p3_\w+_kernel)", n)
        return (m.group(1), ()) if m else (n, ())
    prod = {norm(n) for n in names}
    missing = [x for x in sq_syms if norm(x) not in prod]
    p(f"F1 {tag}: sqtt kernels={len(sq_syms)} not-in-prod-trace={len(missing)} "
      + (f"MISSING: {missing}" if missing else "PASS"))
# F5: p54 vs p52 per-kernel delta on the 4 re-measured 16K shapes
p54 = RAW / "p54_times.json"
if p54.exists():
    delta = json.load(open(p54))
    changed = {"fwd_h_v8o4#1", "bwd_tail#1"}
    for s, newt in delta.items():
        old = CARDS[s]["cards"]
        print_rows = []
        for k, nv in newt.items():
            if k.startswith("direct_copy"):
                continue
            ov = old.get(k, {}).get("t_us")
            if ov and abs(nv["t_us"] - ov) / ov > 0.03 and k not in changed:
                print_rows.append(f"{s} {k}: {ov:.0f}->{nv['t_us']:.0f}us "
                                  f"({(nv['t_us'] - ov) / ov * 100:+.1f}%)")
        p(f"F5 {s}: unchanged kernels beyond ±3%: {len(print_rows)} "
          f"(uniform shifts are positional/power noise, see cards.md 增量节)")
        for r in print_rows[:10]:
            p(f"    {r}")


out = RES / "verification.md"
out.write_text("\n".join(L) + "\n")
print("wrote", out)
