"""`python -m kra predict --machine M --op gemm --shape M=8192,N=4096,K=6144 --dtype bf16`

Shape-only ceiling for one operator: arithmetic intensity, bottleneck, T_theory and the
per-resource lower-bound terms.  Writes PREFIX.predict.json / PREFIX.predict.md when
--out is given, otherwise prints the markdown.  Pure Python; needs no hardware.
"""
from __future__ import annotations

import argparse
import json

from .predict import TheoryMachine, markdown, predict


def _shape(s: str) -> dict:
    out = {}
    for kv in s.split(","):
        k, v = kv.split("=")
        v = v.strip()
        out[k.strip()] = int(v) if v.lstrip("-").isdigit() else float(v)
    return out


def main(argv=None) -> int:
    argv = list(argv if argv is not None else __import__("sys").argv[1:])
    if "--list-ops" in argv:
        from ..ops import registry
        for name, op in sorted(registry().items()):
            print("%-12s shape: %-24s phases: %s" % (name, ",".join(op.params), ",".join(op.phases)))
        return 0

    p = argparse.ArgumentParser(prog="kra predict")
    p.add_argument("--machine", required=True)
    p.add_argument("--op", required=True, help="operator plugin name, e.g. gemm / rmsnorm")
    p.add_argument("--shape", required=True, help="e.g. M=8192,N=4096,K=6144")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--phase", default="fwd", choices=("fwd", "bwd", "fwd+bwd"))
    p.add_argument("--peaks", default="measured", choices=("measured", "theoretical"))
    p.add_argument("--out", default=None)
    p.add_argument("--list-ops", action="store_true", help="list operator plugins and exit")
    a = p.parse_args(argv)

    mach = TheoryMachine.load(a.machine, a.peaks)
    r = predict(mach, a.op, _shape(a.shape), a.dtype, a.phase)
    md = markdown(r)
    if a.out:
        open(a.out + ".predict.json", "w", encoding="utf-8").write(json.dumps(r, indent=1) + "\n")
        open(a.out + ".predict.md", "w", encoding="utf-8").write(md)
        print("wrote %s.predict.json / .md" % a.out)
    else:
        print(md)
    return 0
