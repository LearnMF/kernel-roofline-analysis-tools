"""The multi-round search the methodology prescribes, as a fixed procedure instead of a person.

A single ranked table says which split is best but not *why*, and it cannot say when to stop looking.
The optimisation practice this tool comes from is a loop: look at where the time actually goes, pick the
one factor furthest from its own roof, take the levers that act on that factor, predict each one, adopt
the best, write down what the others cost and why they were refuted, and stop when nothing is left above
the gate.  That loop is what this module does, over the candidates the capacity search already priced.

Round 0 is the unfused implementation -- one kernel per op of the chosen formulation -- because that is
where an implementation actually starts.  A round then evaluates the *neighbours* of the incumbent:

  fuse      two adjacent kernels into one       (removes a handoff from HBM, costs LDS / registers / barriers)
  split     one kernel into two                (relieves capacity: fewer spills, more concurrency)
  reform    the same grouping in another formulation (recompute vs keep the state, op variants)

Every candidate was priced by the same model, so a round's prediction is the model's, not a rule of
thumb; and because the levers are moves in the search space, nothing here is specific to one operator.

Two gates come from the practice, and both are recorded as methodology, not as physics:
  * a predicted gain below GATE_PCT is not a decision this tool may make -- it has to be measured;
  * a lever that leaves every term unchanged is refuted, and the ledger keeps the reason.

The loop reports where it stopped AND what the exhaustive search found, so a greedy path that gets stuck
is visible as such rather than presented as the answer.
"""
from __future__ import annotations

GATE_PCT = 5.0          # methodology: below this a predicted gain has to be confirmed by measurement
MAX_ROUNDS = 12


def _terms(row: dict) -> dict:
    out = {}
    for k in row.get("per_kernel", ()):
        for name, v in k["terms_us"].items():
            out[name] = out.get(name, 0.0) + v
    return out


def _binding_terms(row: dict) -> dict:
    """Only the binding term of each kernel: what the candidate's time is actually made of."""
    out = {}
    for k in row.get("per_kernel", ()):
        out[k["binding"]] = out.get(k["binding"], 0.0) + k["T_us"]
    return out


def diagnose(row: dict, ceiling: dict) -> list:
    """The factors of this candidate's time, largest first, each against its own roof where there is one."""
    roofs = {"memory:" + l["level"]: l["T_roof_us"] for l in ceiling["levels"]}
    out = []
    for name, t in sorted(_binding_terms(row).items(), key=lambda kv: -kv[1]):
        roof = roofs.get(name)
        out.append({"term": name, "T_us": t, "share_pct": 100.0 * t / row["T_split_us"],
                    "roof_us": roof, "over_roof_pct": (100.0 * (t / roof - 1.0)) if roof else None})
    return out


def _shape_of(row: dict) -> list:
    return [frozenset(k.split("+")) for k in row["kernels"]]


def _move(a: list, b: list) -> tuple:
    """(lever, what changed) between two groupings of the same op list, or (None, None) if not adjacent."""
    if len(b) == len(a) - 1:
        for i in range(len(b)):
            if b[i] not in a:
                merged = [x for x in a if x <= b[i]]
                if len(merged) == 2 and a[:i] == b[:i]:
                    return "fuse", " + ".join("+".join(sorted(m)) for m in merged)
        return (None, None)
    if len(b) == len(a) + 1:
        lever, what = _move(b, a)
        return ("split", what) if lever == "fuse" else (None, None)
    if len(b) == len(a) and b != a:
        return (None, None)
    return (None, None)


def _why_worse(inc: dict, cand: dict) -> str:
    """Which terms the refuted move made worse, from the model's own numbers."""
    a, b = _terms(inc), _terms(cand)
    worse = sorted(((b.get(k, 0.0) - a.get(k, 0.0), k) for k in set(a) | set(b)), reverse=True)
    parts = ["%s +%.0f us" % (k, d) for d, k in worse[:3] if d > 0.5]
    return ", ".join(parts) or "no term changed"


def run(by_formulation: dict, ceiling: dict, budget: float | None, max_rounds: int = MAX_ROUNDS) -> dict:
    """by_formulation: {label: [priced rows, feasible and infeasible]}.  Hill-climb with a ledger."""
    rounds, seen, below_from = [], set(), None
    pool = {lab: {tuple(sorted(map(tuple, map(sorted, _shape_of(r))))): r for r in rows if r.get("T_split_us")}
            for lab, rows in by_formulation.items()}
    rejected = {lab: [r for r in rows if not r.get("T_split_us")] for lab, rows in by_formulation.items()}
    # round 0: one kernel per op, in the formulation whose unfused form is cheapest
    starts = [(lab, max(rows.values(), key=lambda r: r["n_kernels"])) for lab, rows in pool.items() if rows]
    if not starts:
        return {"rounds": [], "stop": "no feasible candidate to start from", "gate_pct": GATE_PCT}
    lab, inc = min(starts, key=lambda x: x[1]["T_split_us"])
    rounds.append({"round": 0, "lever": "baseline", "formulation": lab,
                   "kernels": inc["kernels"], "T_split_us": inc["T_split_us"],
                   "gain_pct": 0.0, "verdict": "baseline: one kernel per op, where an implementation starts",
                   "diagnosis": diagnose(inc, ceiling)})
    stop = ""
    while len(rounds) <= max_rounds:
        worst = diagnose(inc, ceiling)[0]
        cands = []
        for olab, rows in pool.items():
            for r in rows.values():
                if r is inc:
                    continue
                if olab == lab:
                    lever, what = _move(_shape_of(inc), _shape_of(r))
                elif _shape_of(r) == _shape_of(inc):
                    lever, what = "reform", olab      # another formulation of the same grouping
                else:
                    lever, what = None, None          # a different op list: not one move away
                if not lever:
                    continue
                cands.append((r["T_split_us"], lever, what, olab, r))
        if not cands:
            stop = "no neighbour of the incumbent is priced: the search space around it is exhausted"
            break
        cands.sort(key=lambda x: x[0])
        best_t, lever, what, olab, best = cands[0]
        gain = 100.0 * (1.0 - best_t / inc["T_split_us"])
        refuted = [{"lever": lv, "what": wt, "T_split_us": t, "gain_pct": 100.0 * (1.0 - t / inc["T_split_us"]),
                    "why": _why_worse(inc, r)} for t, lv, wt, _, r in cands[1:4]]
        for olab2, rows in rejected.items():
            for r in rows[:1]:
                if r.get("why_not"):
                    refuted.append({"lever": "fuse", "what": r["kernels"] and r["kernels"][0],
                                    "T_split_us": None, "gain_pct": None,
                                    "why": "does not fit the chip: " + r["why_not"][:160]})
                break
        key = (olab, tuple(best["kernels"]))
        if gain <= 0.0 or key in seen:
            stop = ("stopped at %.0f us: no neighbour of the incumbent is faster (best was %s %s at "
                    "%+.1f%%).  %s binds and is %s its own roof." % (
                        inc["T_split_us"], lever, what, -gain, worst["term"],
                        "at" if worst["over_roof_pct"] is not None and worst["over_roof_pct"] < GATE_PCT
                        else "above"))
            break
        below = gain < GATE_PCT
        if below and below_from is None:
            below_from = len(rounds)
        seen.add(key)
        inc, lab = best, olab
        rounds.append({"round": len(rounds), "lever": lever, "what": what, "formulation": olab,
                       "kernels": best["kernels"], "T_split_us": best_t, "gain_pct": gain,
                       "below_gate": below,
                       "verdict": ("provisional: %.1f%% is below the %.0f%% gate -- this one has to be "
                                   "measured, not decided here" % (gain, GATE_PCT)) if below else
                                  ("adopted: %s %s, %.1f%% faster, %s now binds"
                                   % (lever, what, gain, diagnose(best, ceiling)[0]["term"])),
                       "refuted": refuted, "mem_peak_bytes": best.get("mem_peak_bytes"),
                       "diagnosis": diagnose(best, ceiling)})
    else:
        stop = "stopped after %d rounds without reaching the gate" % max_rounds
    best_overall = min((r for rows in pool.values() for r in rows.values()), key=lambda r: r["T_split_us"])
    if below_from is not None:
        stop += (("  From round %d on every step is below the %.0f%% gate (%.1f%% of the final time in "
                  "total): that part is a measurement campaign, not a prediction.")
                 % (below_from, GATE_PCT,
                    100.0 * (rounds[below_from - 1]["T_split_us"] / inc["T_split_us"] - 1.0)))
    return {"rounds": rounds, "gate_pct": GATE_PCT, "below_gate_from_round": below_from,
            "reached_us": inc["T_split_us"], "reached": inc["kernels"], "formulation": lab,
            "exhaustive_best_us": best_overall["T_split_us"],
            "greedy_is_exhaustive": abs(inc["T_split_us"] - best_overall["T_split_us"]) < 1e-9,
            "budget_bytes": budget,
            "stop": stop,
            "headroom_to_ceiling_pct": 100.0 * (inc["T_split_us"] / ceiling["T_attain_us"] - 1.0)
            if ceiling.get("T_attain_us") else None}
