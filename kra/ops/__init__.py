"""Operator semantics library: what an operator must compute and must move.

A plugin answers the three questions a machine description cannot:
  flops(shape, phase)        algorithmically required multiply-adds, counted as 2 flop
  bytes_min(shape, phase)    inputs read once + outputs written once, in bytes
  working_set(shape, phase)  bytes that have to stay live on chip for a tile of work
  serial(shape, phase)       dependency structure that cannot be parallelised away
                             (None when the operator is embarrassingly parallel)

Only the last two are specific to this tool: a plain roofline stops at the first two,
which is why it cannot bound chunk-recurrent operators (see docs/sim/design-draft.md).

Operators without a plugin are not estimated.  `get()` raises rather than guessing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

DTYPE_BYTES = {
    "fp64": 8, "f64": 8,
    "fp32": 4, "f32": 4, "tf32": 4,
    "bf16": 2, "f16": 2, "fp16": 2,
    "fp8": 1, "fp8_e4m3": 1, "fp8_block": 1, "i8": 1, "int8": 1,
}

# Which compute peak in machine.peaks a dtype is executed on.
DTYPE_PEAK = {
    "bf16": "mmac_bf16", "f16": "mmac_f16", "fp16": "mmac_f16", "tf32": "mmac_tf32",
    "fp32": "mmac_f32x8", "f32": "mmac_f32x8", "i8": "mmac_i8", "int8": "mmac_i8",
    "fp8": "mmac_fp8", "fp8_e4m3": "mmac_fp8", "fp8_block": "mmac_fp8",
}


def dtype_bytes(dtype: str) -> int:
    try:
        return DTYPE_BYTES[dtype]
    except KeyError:
        raise SystemExit("unknown dtype %r; known: %s" % (dtype, ",".join(sorted(DTYPE_BYTES))))


@dataclass
class Work:
    """The algorithmically required work of one operator instance."""
    flops: float = 0.0
    bytes_min: float = 0.0
    working_set: float = 0.0
    # Dependent steps that cannot overlap each other: [(steps, step_key, ctas), ...].  The
    # per-step latency is a machine property (machine file `structural.recurrence_step_us`).
    serial: list = field(default_factory=list)
    kernels: int = 1
    # When set, HBM is charged as reads at hbm_read + writes at hbm_write instead of
    # bytes_min at the best stream bandwidth.
    bytes_rd: float | None = None
    bytes_wr: float | None = None
    # Machine `structural` key giving the attainable fraction of the compute peak for the
    # operator's tile pattern (e.g. "mmac_tile_eff"); None = the peak itself.
    compute_eff: str | None = None
    # Charge vector-memory issue (machine `structural.vmem_cycles_per_kb`) for the HBM bytes.
    vmem_issue: bool = False
    # Per-resource extra work the operator cannot avoid, e.g. {"sfu_exp2": 1.2e9}.
    other: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    # Kernels that run back to back on one stream.  When set, the bound is the sum of
    # each kernel's own bound (they cannot overlap), which is tighter than bounding the
    # aggregate; the aggregate fields above then hold the totals.
    parts: list = field(default_factory=list)
    # Alternative kernel decompositions of the same work: {name: [(kernel, Work), ...]}.
    # Each gives a T_split; T_theory always bounds the fused work above.
    splits: dict = field(default_factory=dict)

    @classmethod
    def sequence(cls, parts: list) -> "Work":
        """parts: [(name, Work), ...] -> one Work whose totals are the sums."""
        w = cls(flops=sum(p.flops for _, p in parts), bytes_min=sum(p.bytes_min for _, p in parts),
                working_set=max((p.working_set for _, p in parts), default=0.0),
                kernels=sum(p.kernels for _, p in parts), parts=list(parts))
        if all(p.bytes_rd is not None for _, p in parts):
            w.bytes_rd = sum(p.bytes_rd for _, p in parts)
            w.bytes_wr = sum(p.bytes_wr or 0.0 for _, p in parts)
        for _, p in parts:
            w.serial += p.serial
            for k, v in p.other.items():
                w.other[k] = w.other.get(k, 0.0) + v
            w.notes += [n for n in p.notes if n not in w.notes]
        return w


class Op:
    """Base class.  `params` documents the shape keys `kra predict --shape` must supply."""
    name = ""
    params: tuple = ()
    phases: tuple = ("fwd",)

    def work(self, shape: dict, dtype: str, phase: str) -> Work:
        raise NotImplementedError

    def check(self, shape: dict, phase: str) -> None:
        missing = [p for p in self.params if p not in shape]
        if missing:
            raise SystemExit("op %s needs --shape %s (missing %s)"
                             % (self.name, ",".join(self.params), ",".join(missing)))
        if phase not in self.phases:
            raise SystemExit("op %s has no %r phase (has %s)"
                             % (self.name, phase, ",".join(self.phases)))


def registry() -> dict:
    from . import elementwise, gemm, kda
    ops = {}
    for mod in (gemm, elementwise, kda):
        for op in mod.OPS:
            ops[op.name] = op
    return ops


def get(name: str) -> Op:
    ops = registry()
    if name not in ops:
        raise SystemExit("no plugin for operator %r; supported: %s\n"
                         "An operator without a plugin is not estimated -- a number with no "
                         "basis is worse than no number." % (name, ",".join(sorted(ops))))
    return ops[name]
