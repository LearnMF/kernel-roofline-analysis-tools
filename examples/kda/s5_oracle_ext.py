"""Wrapper around the tree's tests/s5/run_fp64_oracle.py (frozen reference_kda.py FP64 oracle, PR #556
criteria) that adds an input variant with harder gates: S5_GATE=extreme scales A_log so that
exp(A_log) reaches ~64 (sigmoid saturates, per-token decays sit at the -5 nat lower bound and the
in-chunk cumulative log2 gates fall far below -126 -- the regime where exp2f's denormal handling and
a bare v_exp_f32 differ).  Everything else (transforms, chunked VJP, tolerances) is the harness's.
    S5_GATE=k3|extreme python3 s5_oracle_ext.py --phase ref|dut T H seed [--arms g2]
Oracle files are tagged by variant: <cache>/<T>_<H>_<seed>[_ext].pt."""
import importlib.util
import os
import sys

import torch

TREE = os.environ.get("S5_TREE", "/public/home/tanbo/g2_opt")
spec = importlib.util.spec_from_file_location("run_fp64_oracle", f"{TREE}/tests/s5/run_fp64_oracle.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

if os.environ.get("S5_GATE", "k3") == "extreme":
    _base = m.base_inputs

    def base_inputs(T, H, seed):
        x = _base(T, H, seed)
        gen = torch.Generator(device="cpu").manual_seed(seed + 1)
        x["A_log"] = torch.log(1.0 + 63.0 * torch.rand(H, generator=gen))   # exp(A_log) in [1, 64]
        return x

    m.base_inputs = base_inputs
    m.CACHE = m.CACHE + "_ext"

if __name__ == "__main__":
    sys.exit(m.main())
