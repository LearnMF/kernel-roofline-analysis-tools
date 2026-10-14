"""Independent test set for the ISA simulator: the compile-time synthetic recurrence steps of rec_step_gen.hip
(ISA from the same standalone binary that was timed, raw/rec_step_gen_v2.txt).
    python3 synth_check.py <rsg.s> rec_step_cfg.txt <rec_step_gen_v2.txt>"""
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isa_sim  # noqa: E402

asm, cfgf, meas = sys.argv[1:4]
text = open(asm).read().splitlines()
sections = {}
cur = None
for l in text:
    m = re.match(r"^(_Z6k_stepI[^:\s]+):", l)
    if m:
        cur = m.group(1)
        sections[cur] = []
    elif cur is not None:
        if l.startswith(".Lfunc_end"):
            cur = None
        else:
            sections[cur].append(l)
by_args = {}
for sym, body in sections.items():
    args = tuple(int(x) for x in re.findall(r"Li(\d+)E", sym))
    by_args[args] = body
mus = {}
for l in open(meas):
    m = re.match(r"STEP (\S+) .* us=([\d.]+)", l)
    if m:
        mus[m.group(1)] = float(m.group(2))
rows = []
for l in open(cfgf):
    if l.startswith("#") or len(l.split()) < 17:
        continue
    t = l.split()
    tag = t[0]
    W, cp, n1, i1, n2, i2, nv1, nv2, nw1, nw2, nr, nld, nst, bar, pf, hbm = map(int, t[1:17])
    key = (W, n1, n2, max(1, i1), nv1, nv2, nw1, nw2, nr, nld, nst, bar, pf)
    if key not in by_args or tag not in mus:
        continue
    with tempfile.NamedTemporaryFile("w", suffix=".s", delete=False) as fo:
        fo.write("\n".join(by_args[key]))
    body = isa_sim.parse_loop(fo.name)
    cyc, _ = isa_sim.simulate(body, W, 16, hbm_frac=1.0 if hbm else 0.0, active_cus=80, ctas_per_cu=cp)
    rows.append((tag, cp, cyc / 1500, mus[tag]))
print("| config | CTAs/CU | simulated us/step | measured | error |")
print("|---|---:|---:|---:|---:|")
errs = []
for tag, cp, p, m in rows:
    errs.append(p / m - 1)
    print(f"| {tag} | {cp} | {p:.3f} | {m:.3f} | {100 * (p / m - 1):+.0f}% |")
inside = sum(abs(e) <= 0.2 for e in errs)
print(f"\n{len(rows)} configs: {inside} within +-20%; median |error| {sorted(abs(e) for e in errs)[len(errs) // 2] * 100:.0f}%")
