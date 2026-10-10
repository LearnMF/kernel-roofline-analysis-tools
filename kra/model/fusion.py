"""Global direction: where the decomposition cost is, as fusion candidates.

From the launcher interface trace (kra.opspec.capture: every launcher call's input/output
tensors with storage pointer and bytes) build the producer -> consumer graph of intermediate
tensors.  For a pair of launchers (P, C), fusing them keeps every tensor P writes and C reads
on chip: its write (if no one else reads it) and C's read leave HBM.  The saving is priced at
the measured attainable bandwidth -- a FLOOR-level estimate (an upper bound on the traffic
saving; on-chip capacity, recompute and occupancy decide whether it is realizable).

    python -m kra.model.fusion --optrace X.optrace.json[.gz] --machine M.json [--top 12]
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path


def _load(p: str) -> dict:
    b = Path(p).read_bytes()
    return json.loads(gzip.decompress(b) if p.endswith(".gz") else b)


def versions(calls: list[dict]) -> list[dict]:
    """Tensor VERSIONS: (producer call, readers) per write.  The caching allocator reuses
    addresses (the forward's freed natives become backward buffers), so a pointer is not a
    tensor identity -- a read belongs to the LAST write of that address before it."""
    last: dict[int, dict] = {}
    out = []
    for i, c in enumerate(calls):
        for t in c.get("inputs") or []:
            v = last.get(t["ptr"])
            if v is not None:
                v["readers"].append(i)
        for t in c.get("outputs") or []:
            v = {"ptr": t["ptr"], "producer": i, "bytes": t["bytes"], "readers": [],
                 "shape": t.get("shape"), "dtype": t.get("dtype")}
            last[t["ptr"]] = v
            out.append(v)
    return out


def candidates(calls: list[dict], bw: float) -> dict:
    pairs = defaultdict(lambda: {"bytes": 0, "tensors": 0, "what": []})
    inter = 0
    for v in versions(calls):
        readers = v["readers"]
        if not readers:
            continue
        i = v["producer"]
        inter += v["bytes"] * (1 + len(readers))
        for j in sorted(set(readers)):
            only = all(r == j for r in readers)
            # fusing (i, j): j's read disappears; i's write too if j is the only reader
            saved = v["bytes"] * (readers.count(j) + (1 if only else 0))
            key = (calls[i]["launcher"], calls[j]["launcher"])
            pairs[key]["bytes"] += saved
            pairs[key]["tensors"] += 1
            pairs[key]["what"].append(f"{v['dtype']}{v['shape']}{'' if only else ' (+other readers)'}")
    ranked = sorted(({"producer": a, "consumer": b, "MB": v["bytes"] / 1e6, "tensors": v["tensors"],
                      "saved_us": v["bytes"] / bw * 1e6, "what": v["what"]} for (a, b), v in pairs.items()),
                    key=lambda x: -x["MB"])
    return {"intermediate_traffic_MB": inter / 1e6, "intermediate_traffic_us": inter / bw * 1e6, "pairs": ranked}


def merged_interface(calls: list[dict], group: set[str]) -> dict:
    """What-if: the launchers in `group` run as ONE kernel (every tensor version produced and
    consumed inside stays on chip).  Interface = external input versions (read once) +
    output versions read outside the group or never read (operator outputs)."""
    vs = versions(calls)
    by_reader: dict[int, list] = defaultdict(list)
    for v in vs:
        for r in v["readers"]:
            by_reader[r].append(v)
    idx = [i for i, c in enumerate(calls) if c["launcher"] in group]
    seen_in, b_in = set(), 0
    for i in idx:
        produced_inside = {id(v) for v in vs if calls[v["producer"]]["launcher"] in group}
        for t in calls[i].get("inputs") or []:
            # the version this read belongs to
            v = next((v for v in by_reader[i] if v["ptr"] == t["ptr"]), None)
            key = id(v) if v is not None else ("ext", t["ptr"])
            if (v is not None and id(v) in produced_inside) or key in seen_in:
                continue
            seen_in.add(key)
            b_in += t["bytes"]
    b_out = 0
    for v in vs:
        if calls[v["producer"]]["launcher"] not in group:
            continue
        outside = [r for r in v["readers"] if calls[r]["launcher"] not in group]
        if outside or not v["readers"]:
            b_out += v["bytes"]
    sep = sum(calls[i]["in_bytes"] + calls[i]["out_bytes"] for i in idx)
    # Convexity: a launcher OUTSIDE the group on a dependency path between two members
    # (e.g. dav -> dhu_bn -> wy_s4_split) makes one fused kernel impossible without
    # recomputing / splitting -- the merge is then only a traffic floor, not a design.
    succ: dict[int, set] = defaultdict(set)
    for v in vs:
        for r in v["readers"]:
            succ[v["producer"]].add(r)

    def reach(a: int) -> set:
        seen, st = set(), [a]
        while st:
            x = st.pop()
            for y in succ[x]:
                if y not in seen:
                    seen.add(y)
                    st.append(y)
        return seen
    members = set(idx)
    between = set()
    for i in idx:
        for k in reach(i) - members:
            if reach(k) & members:
                between.add(calls[k]["launcher"])
    return {"group": sorted(group), "in_MB": b_in / 1e6, "out_MB": b_out / 1e6,
            "merged_MB": (b_in + b_out) / 1e6, "separate_MB": sep / 1e6,
            "convex": not between, "blocked_by": sorted(between)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--optrace", required=True)
    ap.add_argument("--machine", required=True)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--json", default=None)
    ap.add_argument("--merge", action="append", default=[],
                    help="what-if: launchers fused into one, e.g. wy_s4_split+bwd_tail (repeatable)")
    a = ap.parse_args(argv)
    m = json.loads(Path(a.machine).read_text())
    bw = max(m["peaks"][k]["value"] for k in ("hbm_read", "hbm_read_nt", "hbm_write", "hbm_copy") if k in m["peaks"]) * 1e9
    calls = _load(a.optrace)["calls"]
    for g in a.merge:
        w = merged_interface(calls, set(g.split("+")))
        print(f"what-if merge {'+'.join(w['group'])}: interface {w['separate_MB']:,.0f} MB separate -> "
              f"{w['merged_MB']:,.0f} MB merged (in {w['in_MB']:,.0f} + out {w['out_MB']:,.0f}); "
              f"traffic floor {w['separate_MB'] * 1e12 / bw:,.0f} -> {w['merged_MB'] * 1e12 / bw:,.0f} us "
              f"(saves {(w['separate_MB'] - w['merged_MB']) * 1e12 / bw:,.0f} us)"
              + ("" if w["convex"] else f"  [NOT CONVEX: {', '.join(w['blocked_by'])} lies between members]"))
    if a.merge:
        return 0
    r = candidates(calls, bw)
    print(f"intermediate tensors through HBM: {r['intermediate_traffic_MB']:,.0f} MB "
          f"= {r['intermediate_traffic_us']:,.0f} us at {bw / 1e9:,.0f} GB/s")
    print(f"{'producer':14s} -> {'consumer':14s} {'MB saved':>9s} {'us saved':>9s} tensors")
    for p in r["pairs"][:a.top]:
        print(f"{p['producer']:14s} -> {p['consumer']:14s} {p['MB']:9.1f} {p['saved_us']:9.1f} {p['tensors']:7d}")
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
