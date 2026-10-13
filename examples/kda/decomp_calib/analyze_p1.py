#!/usr/bin/env python3
"""P1 analysis: t(residency) and lambda(residency) per kernel.
Reads raw/p1_<T>_<H>/<tag>_trace_*.json collected by p1_collect.sh.

    python3 analyze_p1.py <results_raw_dir>
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_raw import trace_times, short, split_windows, tload  # noqa: E402

TAGMAP = {  # probe tag prefix -> (cards kernel base, lds_target_kb)
    "WU": "wu", "S4A": "wy_s4a", "S4B": "wy_s4b", "DAV": "dav",
    "TAIL": "bwd_tail", "PREPA": "fwd_prep_a", "PREPB": "fwd_prep_b",
    "FWH": "fwd_h_v8o4", "FWHW": "fwd_h_v8o4w",
}


def load_tag_times(d: Path):
    """{tag: {kern#occ: t_us}}"""
    import tempfile
    out = {}
    for f in sorted(d.glob("*_trace_1.json")):
        tag = f.name[: -len("_trace_1.json")]
        # gather this tag's 5 trace files into a temp dir-like view
        recs = {}
        per = {}
        for i in range(1, 6):
            p = d / f"{tag}_trace_{i}.json"
            if not p.exists():
                continue

            cur = {}
            for w in split_windows(tload(p), "spin_kernel", None):
                occ = {}
                for o in w:
                    if o.kind != "kernel":
                        continue
                    n = short(o.name)
                    occ[n] = occ.get(n, 0) + 1
                    cur.setdefault(f"{n}#{occ[n]}", []).append(o.dur_ns / 1e3)
            for k, v in cur.items():
                per.setdefault(k, []).append(v)
        out[tag] = {k: st.median([st.median(v) for v in procs])
                    for k, procs in per.items() if len(procs) >= 3}
    return out


def res_cap(lds_kb: int) -> int:
    return max(1, 65536 // (lds_kb * 1024))


def main(raw: Path, cards: Path):
    C = json.load(open(cards))
    rows = []
    for d in sorted(raw.glob("p1_*_*")):
        m = re.match(r"p1_(\d+)_(\d+)", d.name)
        T, H = int(m.group(1)), int(m.group(2))
        s = f"{T}_{H}"
        c = C.get(s)
        if not c:
            continue
        tags = load_tag_times(d)
        base = tags.get("base", {})
        for tag, times in tags.items():
            if tag == "base":
                continue
            m2 = re.match(r"([A-Z]+)_(\d+)", tag)
            if not m2:
                continue
            pref, kb = m2.group(1), int(m2.group(2))
            for kfull, t in times.items():
                kbase = kfull.split("#")[0]
                if kbase != TAGMAP.get(pref) and not kbase.startswith(
                        TAGMAP.get(pref, "\0") + "w"):
                    continue
                v = c["cards"].get(kfull)
                if not v:
                    continue
                res0, res1 = v["resident"], min(v["resident"], res_cap(kb))
                rnds = v["ctas"] / (80 * res1)
                lam = t / rnds
                tb = base.get(kfull)
                rows.append(dict(shape=s, kernel=kfull, tag=tag, lds_kb=kb,
                                 res0=res0, res1=res1, t_us=t, t_base=tb,
                                 rounds=rnds, lam_us=lam))
    return rows


if __name__ == "__main__":
    raw = Path(sys.argv[1])
    cards = Path(sys.argv[2] if len(sys.argv) > 2 else
                "results/kda-ceiling-cards-20261013/cards.json")
    rows = main(raw, cards)
    for r in rows:
        d = (r["t_us"] - r["t_base"]) / r["t_base"] * 100 if r["t_base"] else 0
        print(f"{r['shape']:9s} {r['kernel']:16s} {r['tag']:10s} res {r['res0']}->{r['res1']} "
              f"t={r['t_us']:8.0f}us ({d:+6.1f}%) lam={r['lam_us']:.1f}us")
    json.dump(rows, open(raw / "p1_rows.json", "w"), indent=1)
