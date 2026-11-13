"""Device-memory liveness of a kernel split: the peak the operator holds in HBM, per rank.

A kernel split is not free in memory.  Every value one kernel produces and a later kernel reads has
to sit in device memory in between, so cutting an operator into more kernels -- or saving a tensor in
the forward instead of recomputing it in the backward -- buys time with capacity.  Training makes the
capacity side a hard constraint: a split whose peak does not fit cannot run at all, however fast its
bound says it is.  Without this term the search always prefers saving over recomputing, because saving
only ever removes work.

This module is operator-independent.  It reads the same capacity graph the split search uses:

  * a value lives from the kernel that produces it to the last kernel that reads it;
  * a value produced and consumed inside one kernel never reaches HBM and costs nothing, except the
    handoffs footprint() had to spill, which are resident for the duration of their kernel;
  * graph inputs are resident throughout (the caller owns them), as are the operator's outputs from
    the kernel that writes them;
  * the peak is the largest total over the stages of the split, not the sum over all values.

The budget is the machine's own `device.global_mem_bytes`, or an explicit per-rank budget when the
caller knows what the rest of the model leaves for this operator.  Nothing is assumed about weights,
optimizer state or other layers: the result is the operator's own peak, reported against whichever
budget was given.
"""
from __future__ import annotations

from .capacity import dataflow


def _values(graph, grouping: list, spilled: list) -> tuple:
    """(persistent, staged): bytes resident throughout, and [(bytes, first_kernel, last_kernel, name)]."""
    src, users = graph._df if hasattr(graph, "_df") else dataflow(graph)
    where = {o.name: i for i, k in enumerate(grouping) for o in k}
    n = len(grouping)
    persistent, staged = [], []
    seen_in = set()
    for k in grouping:
        for o in k:
            for t in o.ins:
                if src[(o.name, t)] is None and t not in seen_in:       # the caller's tensor
                    seen_in.add(t)
                    persistent.append((graph.tile_tensor(t).hbm_bytes, t))
            for t in o.outs:
                tt = graph.tile_tensor(t)
                i = where[o.name]
                readers = [where[c] for c in users.get((o.name, t), ()) if c in where]
                outside = [r for r in readers if r != i]
                if tt.kind == "output":
                    persistent.append((tt.hbm_bytes, t))
                    continue
                if outside:
                    staged.append((tt.hbm_bytes, i, max(outside), t))
                elif (o.name, t) in spilled:
                    staged.append((tt.hbm_bytes, i, i, t))              # round trip inside its kernel
    return persistent, staged


def _fwd_cut(grouping: list) -> int | None:
    """Index of the last kernel of the forward, when the plugin declares a phase per op.  The values live
    across that boundary are the ones the backward reads, i.e. what a save-instead-of-recompute policy
    holds -- the only part of an operator's memory that is multiplied by the number of layers in the model.
    None when the graph is one-sided or the plugin declares no phase (then it is reported as not derived,
    never guessed)."""
    ph = [{o.phase for o in k} for k in grouping]
    if any("" in p for p in ph):
        return None
    fwd = [i for i, p in enumerate(ph) if p == {"fwd"}]
    if not fwd or len(fwd) == len(ph):
        return None
    last = max(i for i in fwd if all(ph[j] == {"fwd"} for j in range(i + 1)))
    return last


def peak(graph, grouping: list, spilled=()) -> dict:
    """Peak device memory of one kernel split, and what is live at the peak."""
    spilled = list(spilled)
    persistent, staged = _values(graph, grouping, spilled)
    base = sum(b for b, _ in persistent)
    stages = []
    for i in range(len(grouping)):
        live = [(b, t) for b, f, l, t in staged if f <= i <= l]
        stages.append({"kernel": "+".join(o.name for o in grouping[i]),
                       "workspace_bytes": sum(b for b, _ in live),
                       "live": [t for _, t in sorted(live, reverse=True)]})
    # Bytes a value holds across a kernel boundary: produced before it, read after it.  Its maximum over
    # the boundaries is what a save-instead-of-recompute policy costs, and in a fwd+bwd graph it is the
    # set carried from the forward into the backward -- the term that scales with the number of layers,
    # while the stage peak below only ever holds one operator's workspace at a time.
    cuts = []
    for i in range(len(grouping) - 1):
        across = [(b, t) for b, f, l, t in staged if f <= i < l]
        cuts.append({"after": "+".join(o.name for o in grouping[i]),
                     "carried_bytes": sum(b for b, _ in across),
                     "carried": [t for _, t in sorted(across, reverse=True)][:8]})
    i = _fwd_cut(grouping)
    saved = None if i is None else {
        "after": cuts[i]["after"], "bytes": cuts[i]["carried_bytes"], "tensors": cuts[i]["carried"]}
    cut = max(cuts, key=lambda c: c["carried_bytes"]) if cuts else {"carried_bytes": 0.0, "carried": [], "after": ""}
    worst = max(stages, key=lambda s: s["workspace_bytes"]) if stages else {"workspace_bytes": 0.0, "live": []}
    return {"peak_bytes": base + worst["workspace_bytes"],
            "persistent_bytes": base,
            "workspace_peak_bytes": worst["workspace_bytes"],
            "at_kernel": worst.get("kernel", ""),
            "saved_for_backward": saved,
            "carried_peak_bytes": cut["carried_bytes"],
            "carried_after": cut["after"],
            "carried_at_peak": cut["carried"],
            "live_at_peak": worst["live"][:8],
            "persistent": [t for _, t in sorted(persistent, reverse=True)][:8],
            "stages": stages}


def resident_bytes(m: dict, layers: int = 1) -> float:
    """What the operator holds when `layers` of it are in flight, as training does: every layer's
    save-for-backward set is live at once, but only one layer's workspace peak is.  layers=1 is the
    operator alone.  Falls back to the plain peak when the plugin declares no phases."""
    saved = (m.get("saved_for_backward") or {}).get("bytes")
    if saved is None or layers <= 1:
        return m["peak_bytes"]
    return m["peak_bytes"] + saved * (layers - 1)


def budget_bytes(mach, explicit: float | None = None) -> tuple:
    """(budget bytes, where it came from).  None when neither is available."""
    if explicit:
        return float(explicit), "given per-rank budget"
    g = float(mach.device.get("global_mem_bytes") or 0)
    if g:
        return g, "device.global_mem_bytes (the whole device: the rest of the model is not subtracted)"
    return None, "machine file has no device.global_mem_bytes: memory is reported but not constrained"


def pareto(rows: list, time_key: str = "T_split_us", mem_key: str = "mem_resident_bytes") -> list:
    """The non-dominated (time, memory) candidates, cheapest first.  A candidate is dominated when
    another is at least as fast AND at least as small."""
    out = []
    for r in sorted(rows, key=lambda r: (r[time_key], r[mem_key])):
        if out and min(o[mem_key] for o in out) <= r[mem_key]:
            continue                                                    # someone faster is also no bigger
        out.append(r)
    return out
