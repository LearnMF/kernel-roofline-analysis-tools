"""Dense GEMM: C[M,N] = A[M,K] @ B[K,N], and its backward pair.

Required flops are exact; required bytes assume each operand crosses HBM once, which is
the optimistic case the lower bound needs (a real kernel re-reads tiles whenever the
working set does not fit -- that cost belongs to T_split, not here).

Backward is two GEMMs, dA[M,K] = dC[M,N] @ B^T and dB[K,N] = A^T @ dC, each its own
kernel, so fwd+bwd is a sequence of three bounds rather than one aggregate.
"""
from __future__ import annotations

from . import Op, Work, dtype_bytes


def _mm(m: int, n: int, k: int, e: int, out_e: int) -> Work:
    """One GEMM producing an [m,n] result from [m,k] x [k,n]."""
    return Work(flops=2.0 * m * n * k, bytes_min=float((m * k + k * n) * e + m * n * out_e),
                working_set=float(min(m, n, k) ** 2 * e))


class Gemm(Op):
    name = "gemm"
    params = ("M", "N", "K")
    phases = ("fwd", "bwd", "fwd+bwd")

    def work(self, shape: dict, dtype: str, phase: str) -> Work:
        self.check(shape, phase)
        m, n, k = (int(shape[x]) for x in ("M", "N", "K"))
        e = dtype_bytes(dtype)
        out_e = max(e, 2)  # fp8/int8 GEMMs accumulate and store at least 16-bit
        fwd = [("fwd C=A@B", _mm(m, n, k, e, out_e))]
        bwd = [("bwd dA=dC@B^T", _mm(m, k, n, out_e, out_e)),
               ("bwd dB=A^T@dC", _mm(k, n, m, out_e, out_e))]
        parts = {"fwd": fwd, "bwd": bwd, "fwd+bwd": fwd + bwd}[phase]
        w = Work.sequence(parts)
        if min(m, n) < 256:
            w.notes.append("min(M,N)=%d: too few output tiles to fill the device; expect the "
                           "fixed-cost floor rather than either roofline to bind" % min(m, n))
        return w


OPS = (Gemm(),)
