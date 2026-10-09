"""Static ISA scans for D-layer causes that PMC/SQTT cannot name (gfx9-class HCU ISA).

D.mathlib_guard: exp2f/expf lowered with a denormal guard (validation P-2, KDA prep_a):
    v_cmp_gt_f32 vcc, <-126>, x ; v_cndmask (bias) ; v_add ; v_exp_f32 ; v_cndmask (scale) ; v_mul
  versus a bare v_exp_f32.  When the argument range is provably >= -126 the guard is dead
  weight (bitwise-identical result); in a VALU-bound kernel removing it saved 17-21%.
"""
from __future__ import annotations

import re
from pathlib import Path

INSN = re.compile(r"^\s+([sv]_[a-z0-9_]+|ds_[a-z0-9_]+|global_[a-z0-9_]+|buffer_[a-z0-9_]+)\b")


def kernel_body(asm: str, symbol: str) -> list[str]:
    lines, out, on = asm.splitlines(), [], False
    for ln in lines:
        if not on and ln.startswith(symbol + ":"):
            on = True
        elif on:
            if ln.startswith(".Lfunc_end"):
                break
            out.append(ln)
    return out


def mathlib_guard(body: list[str], window: int = 6) -> dict:
    ops = [m.group(1) for ln in body if (m := INSN.match(ln))]
    exps = [i for i, o in enumerate(ops) if o.startswith("v_exp_f32")]
    guarded = sum(1 for i in exps
                  if any(o.startswith("v_cmp_gt_f32") or o.startswith("v_cmp_lt_f32")
                         for o in ops[max(0, i - window):i])
                  and any(o.startswith("v_cndmask") for o in ops[i + 1:i + 1 + window]))
    valu = sum(1 for o in ops if o.startswith("v_"))
    est = 5 * guarded
    return {"v_exp": len(exps), "guarded_exp": guarded, "valu_static": valu,
            "guard_insts_est": est, "guard_share_static": est / valu if valu else 0.0}


WAIT = re.compile(r"s_waitcnt\b(.*)")
VMCNT = re.compile(r"vmcnt\((\d+)\)")
LABEL = re.compile(r"^(\.LBB\w+):")
BRANCH = re.compile(r"^\s+s_(?:cbranch_\w+|branch)\s+(\.LBB\w+)")


def load_use_slack(body: list[str]) -> list[dict]:
    """Static exposed-latency finder (validation P-3/P-4): simulate the vector-memory counter
    (gfx9: vmcnt counts loads AND stores, in order) along the text; at every
    `s_waitcnt vmcnt(N)` the ops older than the newest N must have completed.  The slack of a
    wait = instructions issued between the youngest load it forces to complete and the wait.
    A small slack inside a loop means that load's latency is exposed on every iteration --
    the kind of point whose removal paid off (P-3), unlike loads already prefetched (P-4).
    Text order approximates control flow; loop membership comes from backward branches."""
    insns, labels, branches = [], {}, []
    for ln in body:
        if (m := LABEL.match(ln)):
            labels[m.group(1)] = len(insns)
            continue
        if (m := INSN.match(ln)):
            if (b := BRANCH.match(ln)):
                branches.append((len(insns), b.group(1)))
            insns.append(ln.strip())
    loops = [(labels[t], i) for i, t in branches if t in labels and labels[t] <= i]
    q: list[tuple[int, str]] = []
    sites = []
    for i, s in enumerate(insns):
        op = s.split()[0]
        if op.startswith(("global_load", "buffer_load", "flat_load", "global_store", "buffer_store",
                          "flat_store", "global_atomic", "buffer_atomic")):
            q.append((i, op))
            continue
        w = WAIT.match(s)
        if not w or not (vm := VMCNT.search(w.group(1))):
            continue
        n = int(vm.group(1))
        forced = []
        while len(q) > n:
            forced.append(q.pop(0))
        loads = [f for f in forced if "load" in f[1] or "atomic" in f[1]]
        if not loads:
            continue
        young = max(f[0] for f in loads)
        depth = sum(1 for a, b in loops if a <= i <= b)
        sites.append({"insn": i, "wait": s, "slack": i - young - 1, "forced_loads": len(loads),
                      "youngest_load": insns[young], "loop_depth": depth})
    return sites


def scan(asm_path: str | Path, symbol: str) -> dict:
    body = kernel_body(Path(asm_path).read_text(), symbol)
    r = mathlib_guard(body)
    causes = []
    if r["guarded_exp"] and r["guard_share_static"] >= 0.05:
        causes.append({"id": "D.mathlib_guard", "value": round(r["guard_share_static"], 3),
                       "note": f"{r['guarded_exp']} of {r['v_exp']} v_exp_f32 carry a denormal guard "
                               "(~5 VALU each); where the argument is provably >= -126 use "
                               "__builtin_amdgcn_exp2f (bitwise identical in range)"})
    return {"symbol": symbol, "lines": len(body), **r, "causes": causes}
