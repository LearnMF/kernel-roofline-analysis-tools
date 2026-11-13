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
            keys = ",".join(p if isinstance(p, str) else "|".join(p) for p in op.params)
            print("%-12s shape: %-24s phases: %-13s parallel: %s"
                  % (name, keys, ",".join(op.phases), ",".join(op.parallel) or "-"))
        return 0

    p = argparse.ArgumentParser(prog="kra predict")
    p.add_argument("--machine", required=True)
    p.add_argument("--op", required=True, help="operator plugin name, e.g. gemm / rmsnorm")
    p.add_argument("--shape", required=True, help="e.g. M=8192,N=4096,K=6144")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--phase", default="fwd", choices=("fwd", "bwd", "fwd+bwd"))
    p.add_argument("--peaks", default="measured", choices=("measured", "theoretical"))
    p.add_argument("--out", default=None)
    p.add_argument("--split-search", action="store_true",
                   help="solve the capacity-constrained kernel split (needs a plugin capacity graph)")
    p.add_argument("--residency", default="shape", choices=("shape", "lds"),
                   help="how an on-chip handoff is charged: by tensor shape, or everything to LDS")
    p.add_argument("--vgpr", default="full", choices=("full", "acc"),
                   help="register estimate: accumulators + operands + addressing, or accumulators only")
    p.add_argument("--occupancy", default="on", choices=("on", "off"),
                   help="include the concurrency terms (bandwidth attainable at the kernel's CTA count, "
                        "per-CU resources scaled by the CUs it can occupy, memory latency its registers "
                        "cannot hide)")
    p.add_argument("--mem-budget", default=None, type=float,
                   help="device memory this operator may use, per rank, in GB (default: the machine's whole "
                        "device.global_mem_bytes -- the rest of the model is then not subtracted)")
    p.add_argument("--layers", default=1, type=int,
                   help="layers of this operator resident at once: every layer's save-for-backward set is "
                        "live together, only one layer's workspace peak is")
    p.add_argument("--list-ops", action="store_true", help="list operator plugins and exit")
    a = p.parse_args(argv)

    mach = TheoryMachine.load(a.machine, a.peaks, a.occupancy == "on")
    if a.split_search:
        from . import capacity
        rs = capacity.solve(mach, a.op, _shape(a.shape), a.dtype, a.phase, a.residency, vgpr_mode=a.vgpr,
                            mem_budget=a.mem_budget * 2 ** 30 if a.mem_budget else None, layers=a.layers)
        md = capacity.markdown(rs)
        if a.out:
            open(a.out + ".split.json", "w", encoding="utf-8").write(json.dumps(rs, indent=1) + "\n")
            open(a.out + ".split.md", "w", encoding="utf-8").write(md)
            print("wrote %s.split.json / .md" % a.out)
        else:
            print(md)
        return 0
    r = predict(mach, a.op, _shape(a.shape), a.dtype, a.phase)
    md = markdown(r)
    if a.out:
        open(a.out + ".predict.json", "w", encoding="utf-8").write(json.dumps(r, indent=1) + "\n")
        open(a.out + ".predict.md", "w", encoding="utf-8").write(md)
        print("wrote %s.predict.json / .md" % a.out)
    else:
        print(md)
    return 0
