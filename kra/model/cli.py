"""`python -m kra analyze --machine M --pmc P --opspec S --optrace O --shape T=8192,H=12 --out PREFIX`

Builds the per-kernel water-level table (L2 classification + L4 lower bounds) and writes
PREFIX.waterlevel.json / PREFIX.waterlevel.md.  Pure Python; runs anywhere.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bounds import analyze
from .report import to_markdown


def _shape(s: str) -> dict:
    out = {}
    for kv in s.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v) if v.strip().lstrip("-").isdigit() else float(v)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="kra analyze", description=__doc__.splitlines()[0])
    p.add_argument("--machine", required=True)
    p.add_argument("--pmc", required=True, help="PREFIX.pmc.json from `kra pmc`")
    p.add_argument("--opspec", required=True)
    p.add_argument("--optrace", required=True)
    p.add_argument("--shape", required=True, help="e.g. T=8192,H=12")
    p.add_argument("--out", required=True)
    p.add_argument("--title", default=None)
    a = p.parse_args(argv)
    def load(f):
        p = Path(f)
        if p.suffix == ".gz":
            import gzip
            return json.loads(gzip.decompress(p.read_bytes()))
        return json.loads(p.read_text())
    res = analyze(load(a.machine), load(a.pmc), load(a.opspec), load(a.optrace), _shape(a.shape))
    Path(a.out + ".waterlevel.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    Path(a.out + ".waterlevel.md").write_text(to_markdown(res, a.title))
    t = res["totals"]
    print(f"t={t['t_us']:.0f}us prior_bound={t['T_lower_prior_us']:.0f}us "
          f"attainment={t['attainment_prior']:.0%} -> {a.out}.waterlevel.{{json,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
