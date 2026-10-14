"""What-if experiments on the production recurrence step (SQTT loop body) with the validated ISA simulator.
    python3 step_whatif.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isa_sim as S  # noqa: E402

B = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "kda-blocks-20261014", "raw", "sqtt_bodies")
INST = [("fwd MODE 1", "sq_body_mode1.s", 4, 0.267, 48), ("recompute MODE 4", "sq_body_mode4.s", 4, 0.276, 48),
        ("dhu 8-wave", "sq_body_dhu8.s", 8, 0.252, 48)]


def run(body, W, hf, cu):
    return S.simulate(body, W, 16, hbm_frac=hf, active_cus=cu)[0]


def drop(body, pred):
    return [x for x in body if not pred(x)]


def ds_b16_to_b64(body):
    """4 ds_write_b16(_d16_hi) -> 1 ds_write_b64 (+2 v_perm to pack): same LDS bytes, 1/4 of the LDS instructions"""
    out, n = [], 0
    for x in body:
        if x[0].startswith("ds_write_b16"):
            if n % 4 == 0:
                out += [("v_perm_b32", x[1], "valu"), ("v_perm_b32", x[1], "valu"), ("ds_write_b64", x[1], "lds")]
            n += 1
        else:
            out.append(x)
    return out


WHATIF = {
    "no 64-bit address arithmetic (SGPR bases / buffer offsets)": lambda b: drop(b, lambda x: x[2] in ("carry", "valu64")),
    "ds_write_b16 x4 -> ds_write_b64 (+2 perm)": ds_b16_to_b64,
    "half the SALU (2-step unroll)": lambda b: [x for i, x in enumerate(b) if not (x[2] == "salu" and i % 2)],
    "no s_nop": lambda b: drop(b, lambda x: x[2] == "nop"),
}

for name, f, W, hf, cu in INST:
    body = S.parse_loop(os.path.join(B, f))
    base = run(body, W, hf, cu)
    print(f"\n{name}: baseline {base:.0f} cycles ({base / 1500:.3f} us/step)")
    for k, fn in WHATIF.items():
        c = run(fn(body), W, hf, cu)
        print(f"  {k:58s} {c:6.0f}  {100 * (c / base - 1):+5.1f}%")
    allb = body
    for fn in WHATIF.values():
        allb = fn(allb)
    c = run(allb, W, hf, cu)
    print(f"  {'all four':58s} {c:6.0f}  {100 * (c / base - 1):+5.1f}%")
