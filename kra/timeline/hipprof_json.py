"""Loader for DTK hipprof chrome-trace JSON (`hipprof --hip-trace --output-type 0`).

Observed format (DTK 25.10 hipprof):
  * device kernels: ph="X", cat="HIPOPS", tid="Stream<N>", args.BeginNs/EndNs (absolute ns),
    args.devId, args.index (op index), args.procId
  * device copies/memsets: ph="X", cat="HIPCOPY", tid in {SET,H2D,D2H,D2D,...}
  * host API calls: ph="X", cat="HIP", tid="Thread<tid>", args.BeginNs/EndNs, args.index
  * flow events: ph="s" on the API thread (submit side) and ph="t" on the device op,
    sharing `id`; the "t" event's ts equals the op's ts (+1e-4 us).
Only these documented-by-observation fields are used; anything else is ignored.

Timestamp semantics (measured on gfx936 / DTK 25.10, see docs/methodology.md):
a kernel that was already queued behind another one gets BeginNs == previous EndNs, so
device dispatch overhead between back-to-back kernels is folded into the reported
durations and never appears as a gap; tracing also inflates per-dispatch cost
(empty kernel back-to-back: 4.8 us traced vs ~2.0 us unprofiled; isolated: 0.48 us).
Visible gaps therefore mean "the queue ran empty" (host late or synchronization).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DeviceOp:
    kind: str            # "kernel" | "copy"
    name: str
    begin_ns: int
    end_ns: int
    device: int
    stream: str
    index: int
    submit_ns: int | None = None   # host timestamp of the submitting API call (flow "s")
    bytes: int | None = None

    @property
    def dur_ns(self) -> int:
        return self.end_ns - self.begin_ns


@dataclass
class ApiCall:
    name: str
    begin_ns: int
    end_ns: int
    thread: str


@dataclass
class Trace:
    ops: list[DeviceOp] = field(default_factory=list)
    apis: list[ApiCall] = field(default_factory=list)
    source: str = ""


def _ns(args: dict, key: str) -> int | None:
    v = args.get(key)
    return int(v) if v not in (None, "") else None


def load(path: str | Path) -> Trace:
    p = Path(path)
    if p.suffix == ".gz":
        import gzip
        d = json.loads(gzip.decompress(p.read_bytes()))
    else:
        d = json.loads(p.read_text())
    events = d["traceEvents"] if isinstance(d, dict) else d
    tr = Trace(source=str(path))
    # ts (us, relative) -> absolute ns offset, learned from any event carrying both.
    offset = None
    flow_s: dict[int, float] = {}
    flow_t: dict[tuple, int] = {}
    op_by_key: dict[tuple, DeviceOp] = {}
    for e in events:
        ph, cat, args = e.get("ph"), e.get("cat"), e.get("args") or {}
        if ph == "X" and cat in ("HIPOPS", "HIPCOPY", "HIP"):
            b, en = _ns(args, "BeginNs"), _ns(args, "EndNs")
            if b is None or en is None:
                continue
            if offset is None and "ts" in e:
                offset = b - round(float(e["ts"]) * 1000)
            if cat == "HIP":
                tr.apis.append(ApiCall(e.get("name", ""), b, en, str(e.get("tid"))))
                continue
            op = DeviceOp(
                kind="kernel" if cat == "HIPOPS" else "copy",
                name=args.get("name") or e.get("name", ""),
                begin_ns=b, end_ns=en,
                device=int(args.get("devId", 0)),
                # Copies use the copy kind (SET/H2D/...) as tid; the hardware queue id is
                # the common stream identity for kernels and copies.
                stream=f"q{args['queueId']}" if "queueId" in args else str(e.get("tid")),
                index=int(args.get("index", -1)),
                bytes=int(args["Bytes"]) if "Bytes" in args else None,
            )
            tr.ops.append(op)
            op_by_key[(e.get("pid"), round(float(e["ts"]), 3))] = op
        elif ph == "s" and cat == "DataFlow":
            flow_s[e["id"]] = float(e["ts"])
        elif ph == "t" and cat == "DataFlow":
            flow_t[(e.get("pid"), round(float(e["ts"]), 3))] = e["id"]
    if offset is not None:
        for key, fid in flow_t.items():
            op = op_by_key.get(key)
            if op is not None and fid in flow_s:
                op.submit_ns = offset + round(flow_s[fid] * 1000)
    tr.ops.sort(key=lambda o: (o.begin_ns, o.end_ns))
    tr.apis.sort(key=lambda a: a.begin_ns)
    return tr
