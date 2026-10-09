"""`python -m kra pmc --out PREFIX [--marker REGEX] -- <app command...>`

Runs the application twice under DTK hipprof (`--pmc-read`, `--pmc-write`, csv output),
retrying aborted runs (observed intermittently on gfx936 at queue creation), and writes
PREFIX.pmc.json with one record per kernel of the measured window.
`--parse-only` re-parses existing PREFIX_read.csv / PREFIX_write.csv.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from .hipprof_csv import load, window


def _collect(preset: str, prefix: str, app: list[str], retries: int, pause: float) -> Path:
    csv_path = Path(f"{prefix}_{preset}.csv")
    log = Path(f"{prefix}_{preset}.log")
    for attempt in range(1, retries + 1):
        for f in Path(prefix).parent.glob(Path(prefix).name + f"_{preset}.*"):
            f.unlink()
        time.sleep(pause)
        with open(log, "w") as lf:
            subprocess.run(["hipprof", f"--pmc-{preset}", "--pmc-type", "3", "-o", f"{prefix}_{preset}", *app],
                           stdout=lf, stderr=subprocess.STDOUT)
        if csv_path.exists() and csv_path.stat().st_size > 0:
            return csv_path
        print(f"pmc-{preset}: attempt {attempt} produced no data (see {log})")
    raise RuntimeError(f"hipprof --pmc-{preset} failed {retries} times; see {log}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="kra pmc", description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True, help="output prefix")
    p.add_argument("--marker", default=None, help="regex of the marker kernel; keep kernels after the last one")
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--pause", type=float, default=5.0, help="seconds between profiler runs")
    p.add_argument("--parse-only", action="store_true")
    p.add_argument("app", nargs=argparse.REMAINDER)
    a = p.parse_args(argv)
    app = a.app[1:] if a.app[:1] == ["--"] else a.app
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.parse_only:
        rd, wr = Path(f"{a.out}_read.csv"), Path(f"{a.out}_write.csv")
    else:
        if not app:
            p.error("application command required (after --)")
        rd = _collect("read", a.out, app, a.retries, a.pause)
        wr = _collect("write", a.out, app, a.retries, a.pause)
    recs = window(load(rd, wr), a.marker)
    doc = {"schema": "kra.pmc/1", "read_csv": str(rd), "write_csv": str(wr), "marker": a.marker,
           "app": app, "kernels": recs}
    out = Path(f"{a.out}.pmc.json")
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"{len(recs)} kernels -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
