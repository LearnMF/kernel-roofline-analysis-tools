"""`python -m kra timeline TRACE.json [--marker REGEX] [--machine machine.json] ...`

Analyzes a DTK hipprof chrome-trace (`hipprof --hip-trace --output-type 0 -o OUT app`)
and writes OUT.timeline.json + OUT.timeline.md.  Pure Python; can run anywhere.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import analyze as A
from .hipprof_json import load


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="kra timeline", description=__doc__.splitlines()[0])
    p.add_argument("trace", help="hipprof JSON trace")
    p.add_argument("--marker", default=None,
                   help="regex of a marker kernel; each window = ops after one marker up to the next")
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--machine", default=None, help="machine.json from `kra calibrate`")
    p.add_argument("--reference-ms", type=float, default=None,
                   help="unprofiled per-window time (ms) for the trust check")
    p.add_argument("--out", default=None, help="output prefix (default: trace path without .json)")
    p.add_argument("--title", default=None)
    a = p.parse_args(argv)

    machine = json.loads(Path(a.machine).read_text()) if a.machine else None
    res = A.analyze(load(a.trace), a.marker, a.device, machine, a.reference_ms)
    t = Path(a.trace)
    out = Path(a.out) if a.out else t.parent / t.name.split(".json")[0]
    Path(str(out) + ".timeline.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    Path(str(out) + ".timeline.md").write_text(A.to_markdown(res, a.title or f"Timeline (L1): {Path(a.trace).name}"))
    s = res["summary"]
    print(f"windows={res['n_windows']} span_us(median)={s['span_us']['median']:.1f} "
          f"bubble_ratio(median)={s['bubble_ratio']['median']:.2%} -> {out}.timeline.{{json,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
