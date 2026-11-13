"""Capacity-constrained kernel splitting: which kernels an operator must be cut into, and what that costs.

The operator plugin hands over a *capacity graph* -- its ops, the tensors between them, and how much of a
tile of work each needs on chip.  This module is operator-independent: it enumerates the legal ways of
grouping those ops into kernels, drops the groupings whose on-chip working set does not fit the machine,
prices the survivors with the same bound model as `kra.model.predict`, and returns them ranked by T_split.

Two rules make the enumeration legal rather than arbitrary:
  * a *barrier* op (one whose parallel structure differs from its neighbours -- a scan among chunk-parallel
    ops, a reduction over the whole sequence) cannot share a kernel with them: the grid is different;
  * only *contiguous* runs of ops may be grouped, because a kernel executes its ops in order.

Feasibility comes from the machine file (`device.lds_per_cu`, `device.max_vgpr_per_thread`), never from a
measured kernel: the working set is derived from the tensor tiles the ops hold live.  A grouping that does
not fit is reported as infeasible instead of being priced -- that is the term a plain roofline is missing.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from ..ops import Work

VGPR_MAX_DEFAULT = 256          # ISA limit of one thread's architectural vector registers (AMD GCN/CDNA)


@dataclass
class TileTensor:
    """One tensor of the graph, seen twice: as a tile on chip and as traffic in HBM."""
    name: str
    tile_bytes: float               # bytes of this tensor for ONE tile of work (one CTA)
    hbm_bytes: float                # bytes of the whole tensor for the predicted operator instance
    kind: str = "inter"             # input | output | inter  (inputs/outputs always cross HBM)
    residency: str = ""             # "" = whole tile must be on chip | "reg" lane-private | "fold" reformed free
    dtype: str = ""

    def on_chip(self) -> tuple:
        """(lds bytes, per-thread-register bytes) this tensor costs while live inside a kernel."""
        if self.residency == "fold":
            return 0.0, 0.0
        if self.residency == "reg":
            return 0.0, self.tile_bytes
        return self.tile_bytes, 0.0


@dataclass
class CapOp:
    """One op of the operator's dataflow, with the work it must do and the tile it must hold."""
    name: str
    ins: tuple = ()
    outs: tuple = ()
    barrier: bool = False
    phase: str = ""                 # "fwd" | "bwd" | "recompute" | "" when the plugin does not declare it
    core_lds: float = 0.0           # LDS the op needs for its own products (staged operands, whole tiles)
    core_reg_bytes: float = 0.0     # per-CTA register bytes for its accumulators
    movable: tuple = ()             # operand tiles (bytes) that live in registers, or in LDS when registers run out
    ctas: int = 0                   # tiles of work the op's parallel structure gives (its grid)
    flops: float = 0.0
    other: dict = field(default_factory=dict)
    serial: list = field(default_factory=list)


@dataclass
class CapGraph:
    ops: list
    tensors: dict
    threads: int = 256              # threads per CTA of one tile of work
    reg_base_vgpr: int = 0          # per-thread addressing / loop registers every kernel holds
    tiles: int = 0                  # tiles of work in the instance (each stages its core LDS once: W + R)
    compute_eff: str | None = None  # machine `structural` key for the attainable compute fraction
    vmem_issue: bool = False
    note: str = ""

    def tile_tensor(self, name: str) -> TileTensor:
        return self.tensors[name]


# ------------------------------------------------------------------------------------------- enumeration
def segments(graph: CapGraph) -> list:
    """[(kind, [ops]), ...]: barrier ops alone, runs of groupable ops together, in execution order."""
    out, cur = [], []
    for o in graph.ops:
        if o.barrier:
            if cur:
                out.append(("group", cur))
                cur = []
            out.append(("barrier", [o]))
        else:
            cur.append(o)
    if cur:
        out.append(("group", cur))
    return out


def _compositions(seq: list):
    for cuts in itertools.product((0, 1), repeat=max(len(seq) - 1, 0)):
        groups, cur = [], [seq[0]]
        for i, c in enumerate(cuts):
            if c:
                groups.append(cur)
                cur = []
            cur.append(seq[i + 1])
        groups.append(cur)
        yield groups


def groupings(graph: CapGraph):
    """Every legal cut of the graph into kernels: [[op, ...], ...] in execution order."""
    per_seg = [[[s[1]]] if s[0] == "barrier" else list(_compositions(s[1])) for s in segments(graph)]
    for combo in itertools.product(*per_seg):
        yield [k for seg in combo for k in seg]


# ------------------------------------------------------------------------------------------- feasibility
def footprint(graph: CapGraph, kernel: list, vgpr_max: int = VGPR_MAX_DEFAULT,
              lds_cap: float = float("inf")) -> dict:
    """On-chip working set of one kernel: each op's own tile plus the handoffs live across it.

    Placement, in the order a kernel writer falls back:
      operands -- an op's movable operands go to registers first, to LDS when the thread would pass vgpr_max;
      handoffs -- a value passed between ops of the kernel stays on chip (registers / LDS by its shape); when
                  the LDS peak would pass lds_cap, the largest LDS-resident handoff is instead written to
                  global memory by its producer and read back by its consumer inside the same launch (a
                  two-phase kernel).  Spilled handoffs are priced by kernel_work; the fusion still saves the
                  re-read of the inputs the phases share."""
    if not hasattr(graph, "_df"):
        graph._df = dataflow(graph)
    _, users = graph._df
    pos = {o.name: j for j, o in enumerate(kernel)}
    live = []
    for i, o in enumerate(kernel):
        for t in o.outs:
            cons = [pos[c] for c in users.get((o.name, t), ()) if c in pos]
            if cons:
                live.append((t, i, max(cons)))
    reg_cap = (vgpr_max - graph.reg_base_vgpr) * graph.threads * 4.0

    def place(spill):
        lds_pk = reg_pk = 0.0
        moved = 0
        for i, o in enumerate(kernel):
            lds, reg = o.core_lds, o.core_reg_bytes
            for t, p, last in live:
                if (kernel[p].name, t) in spill:
                    continue
                if p <= i <= last and not (p == i and last == i):
                    l, r = graph.tile_tensor(t).on_chip()
                    lds += l
                    reg += r
            for b in sorted(o.movable, reverse=True):
                if reg + b <= reg_cap:
                    reg += b
                else:
                    lds += b
                    moved += 1
            lds_pk, reg_pk = max(lds_pk, lds), max(reg_pk, reg)
        return lds_pk, reg_pk, moved

    spill = set()
    lds_pk, reg_pk, moved = place(spill)
    while lds_pk > lds_cap:
        cand = [(graph.tile_tensor(t).on_chip()[0], kernel[p].name, t) for t, p, last in live
                if p != last and (kernel[p].name, t) not in spill and graph.tile_tensor(t).on_chip()[0] > 0]
        if not cand:
            break
        _, pn, t = max(cand)
        spill.add((pn, t))
        lds_pk, reg_pk, moved = place(spill)
    on_chip = [t for t, p, _ in live if (kernel[p].name, t) not in spill]
    # One barrier per value the block hands over through LDS (or spills to global and reads back):
    # staged operands, LDS-resident handoffs, spilled handoffs.  Register-resident handoffs are
    # lane-private and need none.
    syncs = moved + len(spill) + sum(1 for o in kernel if o.core_lds > 0) \
        + sum(1 for t in on_chip if graph.tile_tensor(t).on_chip()[0] > 0)
    return {"lds_bytes": lds_pk, "vgpr": reg_pk / graph.threads / 4.0 + graph.reg_base_vgpr,
            "handoffs_on_chip": on_chip, "handoffs_spilled": sorted(spill),
            "operands_moved_to_lds": moved, "barriers": syncs}


def feasible(fp: dict, lds_per_cu: float, vgpr_max: int) -> tuple:
    if fp["vgpr"] > vgpr_max:
        return False, "vgpr %.0f > %d" % (fp["vgpr"], vgpr_max)
    if fp["lds_bytes"] > lds_per_cu:
        return False, "lds %.1f KB > %.1f KB" % (fp["lds_bytes"] / 1024, lds_per_cu / 1024)
    return True, ""


def residency(fp: dict, lds_per_cu: float, vgpr_max: int) -> int:
    """Tiles of work resident per CU: how many copies of this working set fit."""
    by_lds = int(lds_per_cu // fp["lds_bytes"]) if fp["lds_bytes"] else 10 ** 6
    by_reg = int(vgpr_max // max(fp["vgpr"], 1.0))
    return max(1, min(by_lds, by_reg))


# ------------------------------------------------------------------------------------------- pricing
def dataflow(graph: CapGraph) -> tuple:
    """Versioned def-use: a tensor name produced twice (a forward value and its backward recompute) is two
    values.  Returns (src, users): src[(op, t)] = the op whose version of t this op reads (None = graph
    input); users[(op, t)] = the ops reading that op's version of t."""
    last, src, users = {}, {}, {}
    for o in graph.ops:
        for t in o.ins:
            p = last.get(t)
            src[(o.name, t)] = p
            if p is not None:
                users.setdefault((p, t), []).append(o.name)
        for t in o.outs:
            last[t] = o.name
    return src, users


def occupancy(graph: CapGraph, kernel: list, fp: dict) -> dict:
    """What the machine solver needs to bound this kernel's concurrency: its grid, its threads, and the
    on-chip footprint one tile holds.  The grid is the parallel structure of the ops in it (they share
    one, because ops of a different structure cannot be grouped)."""
    ctas = min((o.ctas for o in kernel if o.ctas), default=0)
    return {"ctas": ctas, "threads": graph.threads, "lds_bytes": fp["lds_bytes"], "vgpr": fp["vgpr"]}


def kernel_work(graph: CapGraph, kernel: list, produced_elsewhere: dict = None, spilled=()) -> Work:
    """HBM traffic of one kernel: every value it reads that another kernel (or the caller) produced is read
    once, every value it produces that another kernel (or the caller) needs is written once.  Values living
    only inside the kernel cross nothing, except the handoffs footprint() had to spill: those are written
    once and read back once.  Values are versioned (see dataflow)."""
    if not hasattr(graph, "_df"):
        graph._df = dataflow(graph)
    src, users = graph._df
    inside = {o.name for o in kernel}
    rd = wr = 0.0
    reads, writes = [], []
    for o in kernel:
        for t in o.ins:
            p = src[(o.name, t)]
            if p in inside:
                continue
            if (p, t) not in reads:
                reads.append((p, t))
                rd += graph.tile_tensor(t).hbm_bytes
    for o in kernel:
        for t in o.outs:
            tt = graph.tile_tensor(t)
            needed_outside = tt.kind == "output" or any(c not in inside for c in users.get((o.name, t), ()))
            if (needed_outside or (o.name, t) in spilled) and (o.name, t) not in writes:
                writes.append((o.name, t))
                wr += tt.hbm_bytes
            if (o.name, t) in spilled:
                rd += tt.hbm_bytes
    w = Work(flops=sum(o.flops for o in kernel), bytes_rd=rd, bytes_wr=wr,
             compute_eff=graph.compute_eff, vmem_issue=graph.vmem_issue,
             serial=[s for o in kernel for s in o.serial])
    for o in kernel:
        for k, v in o.other.items():
            w.other[k] = w.other.get(k, 0.0) + v
    w.bytes_min = rd + wr
    if graph.tiles:
        lds = sum(o.core_lds for o in kernel) + sum(graph.tile_tensor(t).on_chip()[0]
                                                    for t in footprint(graph, kernel)["handoffs_on_chip"]
                                                    if t not in {x for _, x in spilled})
        w.bytes_lds = 2.0 * lds * graph.tiles or None
    return w


def search(graph: CapGraph, mach, peak_key: str, peak: float, limit: int = 0,
           budget: float | None = None, layers: int = 1) -> dict:
    """Rank every legal grouping by T_split, keeping the infeasible ones visible.

    Two kinds of rejection are reported separately: a grouping whose working set does not fit the chip
    (infeasible), and one that fits but whose device-memory peak exceeds the budget (over budget).  The
    second is what keeps the search from buying time with capacity it does not have."""
    from . import memory
    from .predict import _bound

    dev = mach.device
    lds_per_cu = float(dev.get("lds_per_cu") or dev.get("lds_per_block") or 0)
    if not lds_per_cu:
        raise SystemExit("machine %s has no device.lds_per_cu: the capacity split cannot be solved on it"
                         % mach.id)
    vgpr_max = int(dev.get("max_vgpr_per_thread") or VGPR_MAX_DEFAULT)
    produced = {}
    for o in graph.ops:
        for t in o.outs:
            produced.setdefault(t, []).append(o.name)

    rows = []
    for g in groupings(graph):
        fps = [footprint(graph, k, vgpr_max, lds_per_cu) for k in g]
        verdicts = [feasible(f, lds_per_cu, vgpr_max) for f in fps]
        ok = all(v[0] for v in verdicts)
        row = {"kernels": ["+".join(o.name for o in k) for k in g],
               "n_kernels": len(g), "feasible": ok,
               "lds_peak_kb": max(f["lds_bytes"] for f in fps) / 1024,
               "vgpr_peak": max(f["vgpr"] for f in fps),
               "min_residency": min(residency(f, lds_per_cu, vgpr_max) for f in fps),
               "spilled": ["%s@%s" % (t, p) for f in fps for p, t in f["handoffs_spilled"]],
               "why_not": "; ".join("%s: %s" % (k, v[1]) for k, v in
                                    zip(["+".join(o.name for o in kk) for kk in g], verdicts) if not v[0])}
        if ok:
            parts = []
            for k, f in zip(g, fps):
                w = kernel_work(graph, k, produced, f["handoffs_spilled"])
                w.occupancy = occupancy(graph, k, f)
                w.barriers = (f["barriers"], graph.threads)
                parts.append(("+".join(o.name for o in k), w))
            spilled = [x for f in fps for x in f["handoffs_spilled"]]
            mem = memory.peak(graph, g, spilled)
            row["mem"] = mem
            row["mem_peak_bytes"] = mem["peak_bytes"]
            row["mem_carried_bytes"] = mem["carried_peak_bytes"]
            row["mem_saved_bytes"] = (mem["saved_for_backward"] or {}).get("bytes")
            row["mem_resident_bytes"] = memory.resident_bytes(mem, layers)
            row["within_budget"] = budget is None or row["mem_resident_bytes"] <= budget
            row["occupancy"] = [dict(kernel=n, **(mach.concurrency(w.occupancy) if mach.use_occupancy else {}),
                                     **w.occupancy) for n, w in parts]
            t, krows, _, _ = _bound(mach, parts, peak_key, peak)
            row["T_split_us"] = t * 1e6
            row["bytes"] = sum((w.bytes_rd or 0) + (w.bytes_wr or 0) for _, w in parts)
            row["per_kernel"] = krows
        rows.append(row)
    priced = sorted([r for r in rows if r["feasible"]], key=lambda r: r["T_split_us"])
    feas = [r for r in priced if r["within_budget"]]
    over = [r for r in priced if not r["within_budget"]]
    front = memory.pareto(feas) if feas else []
    front_saved = memory.pareto([r for r in feas if r["mem_saved_bytes"] is not None], mem_key="mem_saved_bytes")
    return {"machine": mach.id, "lds_per_cu": lds_per_cu, "vgpr_max": vgpr_max,
            "candidates": len(rows), "feasible": len(feas), "over_budget": len(over),
            "ranked": feas[:limit] if limit else feas,
            "pareto": front, "pareto_saved": front_saved,
            "smallest": min(feas, key=lambda r: r["mem_resident_bytes"]) if feas else None,
            "over_budget_examples": over[:5],
            "infeasible_examples": [r for r in rows if not r["feasible"]][:5],
            "note": graph.note}


def solve(mach, op_name: str, shape: dict, dtype: str, phase: str,
          residency_mode: str = "shape", top: int = 5, vgpr_mode: str = "full",
          mem_budget: float | None = None, layers: int = 1) -> dict:
    """Operator-independent split solver: ask the plugin for its capacity graphs, search each, and
    return the best feasible kernel split with its T_split.  Operators whose plugin does not expose
    a capacity graph are reported as such -- no split is invented for them."""
    from . import memory
    from ..ops import get

    op = get(op_name)
    if not hasattr(op, "capacity_graphs"):
        raise SystemExit("operator %s has no capacity graph: its plugin declares no dataflow, so the "
                         "kernel split cannot be solved (T_theory is still available)." % op_name)
    peak_key, peak = mach.compute_peak(dtype)
    budget, budget_src = memory.budget_bytes(mach, mem_budget)
    per_form, notes = [], []
    if budget is None:
        notes.append("device memory is reported but NOT constrained: " + budget_src)
    for label, graph in op.capacity_graphs(shape, phase, residency_mode, vgpr_mode):
        r = search(graph, mach, peak_key, peak, budget=budget, layers=layers)
        if not r["ranked"]:
            per_form.append({"formulation": label, "feasible": 0, "candidates": r["candidates"],
                             "over_budget": r.get("over_budget", 0),
                             "why_not": (r["over_budget_examples"][0]["mem"]["at_kernel"] + ": peak %.2f GB > budget"
                                         % (r["over_budget_examples"][0]["mem_resident_bytes"] / 2 ** 30)
                                         if r.get("over_budget_examples") else
                                         r["infeasible_examples"][0]["why_not"] if r["infeasible_examples"] else "")})
            continue
        best = r["ranked"][0]
        per_form.append({"formulation": label, "candidates": r["candidates"], "feasible": r["feasible"],
                         "over_budget": r["over_budget"],
                         "mem": best["mem"], "mem_peak_bytes": best["mem_peak_bytes"],
                         "mem_resident_bytes": best["mem_resident_bytes"],
                         "mem_carried_bytes": best["mem_carried_bytes"],
                         "mem_saved_bytes": best["mem_saved_bytes"],
                         "mem_saved_min_bytes": min([r["mem_saved_bytes"] for r in r["ranked"]
                                                     if r["mem_saved_bytes"] is not None] or [None]),
                         "mem_min_bytes": r["smallest"]["mem_resident_bytes"],
                         "pareto": [{"T_split_us": x["T_split_us"], "mem_resident_bytes": x["mem_resident_bytes"],
                                     "mem_peak_bytes": x["mem_peak_bytes"],
                                     "mem_carried_bytes": x["mem_carried_bytes"], "formulation": label,
                                     "mem_saved_bytes": x["mem_saved_bytes"],
                                     "n_kernels": x["n_kernels"], "kernels": x["kernels"]} for x in r["pareto"]],
                         "T_split_us": best["T_split_us"], "T_worst_us": r["ranked"][-1]["T_split_us"],
                         "kernels": best["kernels"], "lds_peak_kb": best["lds_peak_kb"],
                         "vgpr_peak": best["vgpr_peak"], "min_residency": best["min_residency"],
                         "spilled": best.get("spilled", []), "occupancy": best.get("occupancy", []),
                         "per_kernel": best["per_kernel"], "top": r["ranked"][:top],
                         "note": r["note"]})
    priced = [f for f in per_form if "T_split_us" in f]
    if not priced:
        raise SystemExit("no kernel split of %s runs on %s within the constraints: every grouping either "
                         "exceeds the on-chip capacity or exceeds the device-memory budget (%s).  The "
                         "operator cannot run on this machine as described." % (op_name, mach.id, budget_src))
    best = min(priced, key=lambda f: f["T_split_us"])
    cands = [x for f in priced for x in f["pareto"]]
    front = memory.pareto(cands)
    front_saved = memory.pareto([x for x in cands if x["mem_saved_bytes"] is not None],
                                mem_key="mem_saved_bytes")
    return {"schema": "kra.split_search/1", "machine": mach.id, "op": op_name, "phase": phase,
            "mem_budget_bytes": budget, "mem_budget_source": budget_src, "layers": layers,
            "mem_resident_bytes": best["mem_resident_bytes"],
            "mem_peak_bytes": best["mem_peak_bytes"],
            "mem_carried_bytes": best["mem_carried_bytes"],
            "mem_saved_bytes": best["mem_saved_bytes"],
            "mem_min_bytes": min(f["mem_min_bytes"] for f in priced),
            "pareto": front, "pareto_saved": front_saved,
            "shape": dict(shape), "residency_mode": residency_mode, "vgpr_mode": vgpr_mode,
            "lds_per_cu_kb": float(mach.device.get("lds_per_cu", 0)) / 1024,
            "best": best, "formulations": per_form, "notes": notes}


def _gb(b) -> str:
    return "-" if b is None else "%.3f GB" % (b / 2 ** 30)


def markdown(r: dict) -> str:
    b = r["best"]
    L = ["# kernel split search: %s %s on %s" % (r["op"], r["phase"], r["machine"]), "",
         "shape `%s`, LDS %.0f KB/CU, residency rule `%s`, vgpr rule `%s`" % (
             ",".join("%s=%s" % kv for kv in r["shape"].items()), r["lds_per_cu_kb"], r["residency_mode"],
             r.get("vgpr_mode", "full")), "",
         "device memory: budget %s (%s), %d layer(s) of this operator resident" % (
             _gb(r["mem_budget_bytes"]), r["mem_budget_source"], r.get("layers", 1)), "",
         "| formulation | candidates | feasible | over budget | best T_split | worst feasible | kernels | "
         "peak mem | saved for bwd | LDS peak | VGPR peak | tiles/CU |",
         "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for f in r["formulations"]:
        if "T_split_us" not in f:
            L.append("| %s | %d | 0 | %d | - | - | - | - | - | - | - | - |  <- %s" % (
                f["formulation"], f["candidates"], f.get("over_budget", 0), f.get("why_not", "")))
            continue
        L.append("| %s | %d | %d | %d | **%.0f us** | %.0f us | %d | %s | %s | %.1f KB | %.0f | %d |" % (
            f["formulation"], f["candidates"], f["feasible"], f["over_budget"], f["T_split_us"],
            f["T_worst_us"], len(f["kernels"]), _gb(f["mem_peak_bytes"]), _gb(f["mem_saved_bytes"]),
            f["lds_peak_kb"], f["vgpr_peak"], f["min_residency"]))
    mem = b["mem"]
    L += ["", "## device memory of that split", "",
          "the stage peak is one instance of the operator; the save-for-backward set is what every layer of "
          "the model holds at once, so it is the term a recompute policy trades against time.", "",
          "| term | bytes | where |", "| --- | ---: | --- |",
          "| caller's tensors (inputs, outputs, grads) | %s | resident throughout |" % _gb(mem["persistent_bytes"]),
          "| workspace at its peak | %s | at `%s`: %s |" % (
              _gb(mem["workspace_peak_bytes"]), mem["at_kernel"], ", ".join("`%s`" % t for t in mem["live_at_peak"])),
          "| **stage peak** | **%s** | |" % _gb(mem["peak_bytes"]),
          "| saved for the backward | %s | %s |" % (
              _gb((mem["saved_for_backward"] or {}).get("bytes")),
              ("after `%s`: %s" % ((mem["saved_for_backward"] or {}).get("after", ""),
                                   ", ".join("`%s`" % t for t in (mem["saved_for_backward"] or {}).get("tensors", ())))
               if mem["saved_for_backward"] else "not derived: the plugin declares no phase per op")),
          "| resident at %d layer(s) | %s | against the budget |" % (
              r.get("layers", 1), _gb(b["mem_resident_bytes"]))]
    for key, axis, head in (("pareto", "mem_resident_bytes", "time vs resident memory"),
                            ("pareto_saved", "mem_saved_bytes", "time vs bytes saved for the backward")):
        front = r.get(key) or []
        if len(front) < 2:
            continue
        L += ["", "## %s: the non-dominated candidates" % head, "",
              "| T_split | memory | formulation | kernels |", "| ---: | ---: | --- | ---: |"]
        L += ["| %.0f us | %s | %s | %d |" % (x["T_split_us"], _gb(x[axis]), x["formulation"], x["n_kernels"])
              for x in front]
    L += ["", "## best split: %s, T_split = %.3f ms" % (b["formulation"], b["T_split_us"] / 1e3), "",
          "| kernel | binding | T_us |", "| --- | --- | ---: |"]
    L += ["| %s | %s | %.1f |" % (k["kernel"], k["binding"], k["T_us"]) for k in b["per_kernel"]]
    occ = [o for o in b.get("occupancy", []) if o.get("waves_per_cu")]
    if occ:
        L += ["", "## occupancy of that split", "",
              "concurrency the kernel's own footprint allows, and what limits it (LDS / registers / wave "
              "slots / grid).  Per-CU resources are charged at the CUs the kernel can occupy; its reads at "
              "the bandwidth measured at that CTA count.", "",
              "| kernel | CTAs | LDS/tile | VGPR | waves/CU | wave slots | limited by | CUs used |",
              "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |"]
        L += ["| %s | %d | %.1f KB | %.0f | %d | %.0f%% | %s | %d |" % (
            o["kernel"], o["ctas"], o["lds_bytes"] / 1024, o["vgpr"], o["waves_per_cu"],
            o["wave_slots_pct"], o["occupancy_limit"], o["cus_used"]) for o in occ]
    if b.get("spilled"):
        L += ["", "handoffs that do not fit on chip and round-trip through global memory inside their kernel: "
              + ", ".join("`%s`" % x for x in b["spilled"])]
    L += ["", b["note"], ""]
    return "\n".join(L)


def rank_of(res: dict, kernels: list) -> int | None:
    """1-based rank of a named grouping in the ranked list (for hold-out validation)."""
    want = [set(k.split("+")) for k in kernels]
    for i, r in enumerate(res["ranked"], 1):
        if [set(k.split("+")) for k in r["kernels"]] == want:
            return i
    return None
