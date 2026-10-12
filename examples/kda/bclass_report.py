"""Report of the B-class precision gate (R23b rules 1-3) from bclass_gate.sh output.
  rule 1  determinism: every candidate run reports deterministic=true and finite=true
  rule 2  FP64 accuracy not worse than production: per tensor, geometric mean of cand/prod rms_ratio
          <= 1 + eps (default 0.01) AND the paired sign test does not show cand significantly worse
          (one-sided p < 0.05 fails)
  rule 3  within FLA: every case passes PR #556 tolerances, and per tensor the median cand/FLA
          rms ratio <= 1.10 (FLA on the same inputs and hardware)
    python3 bclass_report.py out.jsonl [--eps 0.01]"""
import argparse
import collections
import json
import math
import statistics as st


def comb(n, k):   # math.comb needs Python 3.8; bw47's host python is older
    r = 1
    for j in range(1, k + 1):
        r = r * (n - k + j) // j
    return r

ap = argparse.ArgumentParser()
ap.add_argument("jsonl")
ap.add_argument("--eps", type=float, default=0.01)
a = ap.parse_args()
rows = [json.loads(l) for l in open(a.jsonl) if l.strip()]
by = collections.defaultdict(dict)
for r in rows:
    if r["res"] is None:
        print("MISSING run:", r["role"], r["gate"], r["T"], r["H"], r["seed"])
        continue
    by[(r["gate"], r["T"], r["H"], r["seed"])][r["role"]] = r["res"]
ok = True
det_bad = [k for k, v in by.items() if "cand" in v and not (v["cand"]["g2"]["deterministic"] and v["cand"]["g2"]["finite"])]
print(f"rule 1 determinism/finite: {'PASS' if not det_bad else 'FAIL ' + str(det_bad)}")
ok &= not det_bad
names = None
ratio, worse, fla_r, tol_bad = collections.defaultdict(list), collections.Counter(), collections.defaultdict(list), []
for k, v in by.items():
    if "cand" not in v or "prod" not in v:
        continue
    c, p = v["cand"]["g2"], v["prod"]["g2"]
    f = v["prod"].get("fla")
    names = list(c["rms_ratio"])
    if not all(c["pass"].values()):
        tol_bad.append((k, [n for n, x in c["pass"].items() if not x]))
    for n in names:
        rr = c["rms_ratio"][n] / max(p["rms_ratio"][n], 1e-30)
        ratio[n].append(rr)
        worse[n] += rr > 1
        if f is not None:
            fla_r[n].append(c["rms_ratio"][n] / max(f["rms_ratio"][n], 1e-30))
print("rule 2 FP64 accuracy vs production (cand/prod rms_ratio over paired cases):")
for n in names or []:
    xs = ratio[n]
    gm = math.exp(sum(math.log(x) for x in xs) / len(xs))
    N, k = len(xs), worse[n]
    p_one = sum(comb(N, i) for i in range(k, N + 1)) / 2 ** N   # P(>= k worse | no difference)
    good = gm <= 1 + a.eps and p_one >= 0.05
    ok &= good
    print(f"   {n:9s} geo-mean {gm:.4f}  range [{min(xs):.3f},{max(xs):.3f}]  worse {k}/{N}  one-sided p {p_one:.3f}  {'ok' if good else 'FAIL'}")
print(f"rule 3a PR#556 tolerances on every candidate case: {'PASS' if not tol_bad else 'FAIL ' + str(tol_bad)}")
ok &= not tol_bad
if fla_r:
    print("rule 3b vs FLA on the same inputs (median cand/FLA rms_ratio <= 1.10):")
    for n in names:
        md = st.median(fla_r[n])
        good = md <= 1.10
        ok &= good
        print(f"   {n:9s} median {md:.3f}  max {max(fla_r[n]):.3f}  {'ok' if good else 'FAIL'}")
print("B-CLASS GATE (rules 1-3):", "PASS" if ok else "FAIL", "  (rule 4 = Megatron loss alignment, separate)")
