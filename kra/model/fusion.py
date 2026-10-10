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


def candidates(calls: list[dict], bw: float) -> dict:
    prod, cons, size = {}, defaultdict(list), {}
    for i, c in enumerate(calls):
        for t in c.get("inputs") or []:
            cons[t["ptr"]].append(i)
            size.setdefault(t["ptr"], t["bytes"])
        for t in c.get("outputs") or []:
            prod.setdefault(t["ptr"], i)
            size[t["ptr"]] = t["bytes"]
    pairs = defaultdict(lambda: {"bytes": 0, "tensors": 0})
    inter = 0
    for ptr, i in prod.items():
        readers = [j for j in cons.get(ptr, []) if j > i]
        if not readers:
            continue
        inter += size[ptr] * (1 + len(readers))
        for j in set(readers):
            only = all(r == j for r in readers)
            # fusing (i, j): j's read disappears; i's write too if j is the only reader
            saved = size[ptr] * (readers.count(j) + (1 if only else 0))
            key = (calls[i]["launcher"], calls[j]["launcher"])
            pairs[key]["bytes"] += saved
            pairs[key]["tensors"] += 1
    ranked = sorted(({"producer": a, "consumer": b, "MB": v["bytes"] / 1e6, "tensors": v["tensors"],
                      "saved_us": v["bytes"] / bw * 1e6} for (a, b), v in pairs.items()),
                    key=lambda x: -x["MB"])
    return {"intermediate_traffic_MB": inter / 1e6, "intermediate_traffic_us": inter / bw * 1e6, "pairs": ranked}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--optrace", required=True)
    ap.add_argument("--machine", required=True)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    m = json.loads(Path(a.machine).read_text())
    bw = max(m["peaks"][k]["value"] for k in ("hbm_read", "hbm_read_nt", "hbm_write", "hbm_copy") if k in m["peaks"]) * 1e9
    r = candidates(_load(a.optrace)["calls"], bw)
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
