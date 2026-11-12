"""Elementwise and row-reduction operators: the memory-bound end of the range.

`elements` is the number of elements of the *output* tensor; `streams` is how many
tensors of that size cross HBM (reads + writes).  RMSNorm fwd for one tensor is
streams=2 (read x, write y) plus the row statistic, which is a second read of x unless
the row fits on chip -- that choice is a T_split question, so the bound assumes one pass.
"""
from __future__ import annotations

from . import Op, Work, dtype_bytes


class Elementwise(Op):
    name = "elementwise"
    params = ("elements",)
    phases = ("fwd", "bwd", "fwd+bwd")

    def work(self, shape: dict, dtype: str, phase: str) -> Work:
        self.check(shape, phase)
        n = float(shape["elements"])
        e = dtype_bytes(dtype)
        streams = float(shape.get("streams", 2))
        flops_per = float(shape.get("flops_per_element", 1))
        sfu_per = float(shape.get("sfu_per_element", 0))
        if phase in ("bwd", "fwd+bwd"):
            # The backward pass re-reads the forward input alongside the gradient.
            streams += float(shape.get("bwd_streams", 3))
            flops_per *= 2
        w = Work(flops=n * flops_per, bytes_min=n * streams * e, working_set=0.0)
        if sfu_per:
            w.other["sfu_exp2"] = n * sfu_per
        w.notes.append("streams=%g tensors of %g elements cross HBM" % (streams, n))
        return w


class Rmsnorm(Elementwise):
    name = "rmsnorm"

    def work(self, shape: dict, dtype: str, phase: str) -> Work:
        s = dict(shape)
        s.setdefault("streams", 2)
        s.setdefault("flops_per_element", 4)
        return super().work(s, dtype, phase)


class Softmax(Elementwise):
    name = "softmax"

    def work(self, shape: dict, dtype: str, phase: str) -> Work:
        s = dict(shape)
        s.setdefault("streams", 2)
        s.setdefault("flops_per_element", 3)
        s.setdefault("sfu_per_element", 1)
        return super().work(s, dtype, phase)


OPS = (Elementwise(), Rmsnorm(), Softmax())
