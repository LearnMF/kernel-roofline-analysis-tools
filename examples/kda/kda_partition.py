"""Capacity pre-pruning of the chunk-parallel BACKWARD groupings (phase 2, before the constructive cost model).

The backward's chunk-parallel ops (one (chunk, head) tile per CTA, the same grid) sit in segments separated by the
two scans (kda_ir.formulation order):  S0 = before the scans, S1 = between them, S2 = after both.  A kernel can only
group ops of one segment (a scan is a hard boundary, R48 / C1).  For every composition of every segment into
contiguous kernels this script reports
  * HBM handoff bytes per token*head (tensors produced in one kernel and consumed by another, write + read per
    consuming kernel; tensors needed by another segment are always written),
  * the on-chip working set of each kernel:
        LDS  = max over its ops i of  core_lds(i) + sum of the on-chip handoffs live across i  (mode lds / tile)
        VGPR = max over its ops i of  core_vgpr(i) + sum of the register handoffs live across i  (mode reg)
    core_* = the measured LDS / VGPR of the production kernel that runs the op today (ceiling cards); this is the
    footprint of the CURRENT op implementations, not a minimum -- a redesigned op could be smaller,
  * residency (64 KB LDS / CU; VGPR rule of R38: 768 / VGPR waves per SIMD) and a verdict:
        infeasible (VGPR > 256 or LDS > 64 KB) | res1 (fits, 1 CTA / CU: price with the occupancy curves) | res2+.
No timing here: bytes saved are NOT time saved (R36); the constructive model prices the survivors.
    python3 kda_partition.py [--dav split|whole] [--C 64] [--budget] [--scenarios]"""
import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import kda_ir as ir  # noqa: E402

C, THREADS = 64, 256
LDS_CU = 65536

# measured core footprints of the production kernels running each op (16K cards; identical at 16K/48)
CORE = {   # op: (lds bytes, vgpr, production kernel)
    "re_l2norm": (0, 20, "l2n_apply"),
    "re_gate": (2048, 96, "gate"),
    "re_wy": (26880, 212, "wu"),
    "dav": (17408, 76, "dav"),
    "dav_v": (17408, 76, "dav (pass 2 only; the whole dav footprint, conservative)"),
    "dav_a": (17408, 76, "dav (pass 1 only; the whole dav footprint, conservative)"),
    "wy_bwd_a": (30464, 152, "wy_s4a"),
    "wy_bwd_b": (32768, 236, "wy_s4b"),
    "tail": (6144, 120, "bwd_tail"),
}

# LDS buffers of the production op implementations, as functions of the chunk size c (source: g2_wu.cuh,
# g2_dav.cuh, g2_wy.cuh, g2_bwd_tail.cuh; at c = 64 they add up to the measured footprints byte for byte).
#   need : data exchange between waves / a layout change that must go through LDS (bitwise: no K-permutation
#          tricks, R44)
#   cache: data reused from LDS that could be re-read from L1 / L2 instead (more VMEM / TA)
#   pipe : double buffers / prefetch stages (latency hiding)
#   small: scalars and per-row vectors
BUFFERS = {
    "re_wy": [("slab: transposed k / v staging for the B operand (union with the row-major gated tile)",
               lambda c: 128 * (c + 4) * 2, "need"),
              ("sA: T^-1 kept for both MMAC phases", lambda c: c * (c + 4) * 2, "cache"),
              ("sBeta + sGl", lambda c: c * 4 + 128 * 4, "small")],
    "dav": [("slab union: pass-1 B staging [c][132] / dA fp32 [c][c+4] / pass-2 [128][c+4]",
             lambda c: max(c * 132 * 2, c * (c + 4) * 4, 128 * (c + 4) * 2), "need")],
    "wy_bwd_a": [("S: dv^T -> dw^T transpose", lambda c: 128 * (c + 4) * 2, "need"),
                 ("XT: dA-stage staging", lambda c: c * (c + 4) * 2, "need"),
                 ("KGs: kgB tile of step kt, double-buffered (P-28)", lambda c: 2 * 16 * c * 2, "pipe"),
                 ("BETAs", lambda c: c * 4, "small")],
    "wy_bwd_b": [("TA / TQ / TK tables (16 k x c rows, read by every wave)", lambda c: 3 * 16 * (c + 4) * 4, "need"),
                 ("gM + cs1 / cs2", lambda c: (c // 16) * 16 * 4 + 2 * 2 * (c // 16) * 16 * 4, "small"),
                 ("stg: h / dh tiles of step kt, single buffer (16 x 128 bf16 each, c-independent)",
                  lambda c: 2 * 16 * 128 * 2, "need"),
                 ("stg: second buffer (prefetch one kt ahead, P4'')", lambda c: 2 * 16 * 128 * 2, "pipe"),
                 ("dwTs: dwT tile of step kt (P-29)", lambda c: 16 * c * 2, "need")],
    "tail": [("sDA / sBlk / sDDt partials", lambda c: 3 * 8 * c * 4, "need")],
    "re_gate": [("gate smem", lambda c: 2048 * c // 64, "need")],
    "re_l2norm": [],
}
BUFFERS["dav_v"] = BUFFERS["dav_a"] = BUFFERS["dav"]


def core_lds(op, c, lean=False):
    keep = ("need", "small") if lean else ("need", "small", "cache", "pipe")
    return sum(f(c) for _, f, k in BUFFERS.get(op, []) if k in keep)


# how a tensor handed from one op to the next INSIDE one kernel has to live on chip (per CTA, C = 64)
#   fold : the consumer forms it from what it loads anyway (l2norm with the saved rstd): 0 B, extra VALU
#   whole: the consumer needs the complete per-CTA tensor before it starts (LDS)
#   tile : streamed tile by tile (double-buffered tile in LDS)
#   reg  : same lane mapping on both sides (registers)
EDGE = {
    "qn": ("fold", "re_l2norm -> re_wy: bf16(q*rstd) at load time (P-13 measured wu +10% when done in wu)"),
    "kn": ("fold", "as qn"),
    "g": ("whole", "column-wise chunk cumsum, read by wu at arbitrary (row, col) in several phases: fp32 64x128"),
    "dAqk": ("whole", "s4b reads its rows (aqA) and columns (aqB): fp32 64x64"),
    "dA": ("whole", "s4b reads rows (dak) and columns (akB): fp32 64x64"),
    "dwT": ("whole", "s4a's pass W produces all of it before s4b's kt loop: bf16 128x64"),
    "dq": ("whole", "tail's l2norm backward needs the full-row dot before applying it: fp32 64x128"),
    "dk": ("whole", "as dq"),
    "dg": ("tile", "reverse cumsum over rows per column: one 16-column fp32 tile per kt step, double-buffered"),
    "db": ("reg", "per-row partial sums"),
}
TILE_FRAC = 2 / 8     # 2 buffers of one kt tile (16 of 128 columns)


def onchip_bytes(t):
    mode = EDGE.get(t, ("whole", ""))[0]
    full = ir.T_[t].elems_per_tok(C) * C * ir.DT[ir.MIN_DT.get(t, "bf16")]   # per CTA = one chunk of C rows
    return mode, {"fold": 0, "whole": full, "tile": full * TILE_FRAC, "reg": full}[mode]


def segments(dav_mode):
    ops = [o for o in ir.formulation() if o.phase in ("bwd", "recompute")]
    if dav_mode == "split":
        new = []
        for o in ops:
            if o.name == "dav":
                new.append(ir._with(o, name="dav_v", ins=("do", "Aqk"), outs=("dVn",), tri=1))
                new.append(ir._with(o, name="dav_a", ins=("do", "v_new"), outs=("dAqk",), tri=1))
            else:
                new.append(o)
        ops = new
    segs, cur, scans = [], [], []
    produced_by_scan = set()
    for o in ops:
        if o.par.startswith("scan"):
            segs.append(cur)
            cur = []
            scans.append(o)
            produced_by_scan |= set(o.outs)
        elif o.par in ("chunk", "row"):
            cur.append(o)
    segs.append(cur)
    if dav_mode == "split":   # dav_v needs no scan output: it belongs to S0 (dVn feeds scan_dh)
        for s in segs[1:]:
            for o in list(s):
                if o.name == "dav_v":
                    s.remove(o)
                    segs[0].append(o)
        # dav_a needs v_new (re_scan) and is not needed by scan_dh: S2
        for o in list(segs[1]):
            if o.name == "dav_a":
                segs[1].remove(o)
                segs[2].insert(0, o)
    return segs, ops


def compositions(seq):
    n = len(seq)
    for cuts in itertools.product((0, 1), repeat=max(n - 1, 0)):
        groups, cur = [], [seq[0]]
        for i, c in enumerate(cuts):
            if c:
                groups.append(cur)
                cur = []
            cur.append(seq[i + 1])
        groups.append(cur)
        yield groups


CORE_MODE = {"lean": False}


def kernel_footprint(group):
    names = [o.name for o in group]
    live = []    # (tensor, producer index, last consumer index)
    for i, o in enumerate(group):
        for t in o.outs:
            cons = [j for j, c in enumerate(group) if j > i and t in c.ins]
            if cons:
                live.append((t, i, max(cons)))
    lds_pk = vgpr_pk = 0
    for i, o in enumerate(group):
        l, v, _ = CORE[o.name]
        if CORE_MODE["lean"] or C != 64:
            l = core_lds(o.name, C, lean=CORE_MODE["lean"])
        for t, p, last in live:
            if p <= i <= last and not (p == i and last == i):
                mode, b = onchip_bytes(t)
                if mode == "reg":
                    v += b / THREADS / 4
                else:
                    l += b
        lds_pk, vgpr_pk = max(lds_pk, l), max(vgpr_pk, v)
    waves_simd = 768 // max(int(vgpr_pk), 1)
    res = min(LDS_CU // max(int(lds_pk), 1), waves_simd) if lds_pk else waves_simd
    if vgpr_pk > 256 or lds_pk > LDS_CU:
        verdict = "infeasible"
    elif res <= 1:
        verdict = "res1"
    else:
        verdict = f"res{min(res, 3)}"
    return dict(ops=names, lds=lds_pk, vgpr=vgpr_pk, res=res, verdict=verdict,
                onchip=[t for t, p, last in live])


def handoff_bytes(all_ops, kernels):
    """HBM bytes of every tensor produced by one backward chunk kernel and consumed by another kernel (any
    segment) or by a scan: write once + one read per consuming kernel / scan."""
    where = {}
    for ki, k in enumerate(kernels):
        for o in k:
            where[o.name] = ki
    tot = 0.0
    detail = {}
    for o in all_ops:
        if o.name not in where:
            continue
        for t in o.outs:
            if ir.T_[t].kind in ("output", "grad_param"):
                continue
            readers = set()
            for c in all_ops:
                if t in c.ins and c is not o:
                    readers.add(where.get(c.name, "scan:" + c.name))
            readers.discard(where[o.name])
            if readers:
                b = ir.T_[t].elems_per_tok(C) * ir.DT[ir.MIN_DT.get(t, "bf16")]
                detail[t] = b * (1 + len(readers))
                tot += detail[t]
    return tot, detail


HISTORY = {
    ("wy_bwd_a", "wy_bwd_b"): "F6 (P-39): LDS does not fit -> slower",
    ("wy_bwd_b", "tail"): "B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x",
    ("re_l2norm", "re_wy"): "F1 / P-36: l2n folded into wu, wu +22-27%",
}


def budget(groups):
    """LDS composition of selected fused groups: core of the largest op vs the on-chip handoffs, and the gap to
    the 2-CTA/CU budget (32 KB)."""
    segs, ops = segments("split")
    byname = {o.name: o for o in ops}
    print("| fused group | LDS peak KB | largest op core KB | on-chip handoffs (KB) | gap to 32 KB (res 2) |")
    print("|---|---:|---:|---|---:|")
    for g in groups:
        k = [byname[n] for n in g]
        fp = kernel_footprint(k)
        core = max(CORE[n][0] for n in g) / 1024
        ho = ", ".join(f"{t} {onchip_bytes(t)[1] / 1024:.0f}" for t in fp["onchip"] if onchip_bytes(t)[1])
        print(f"| {'+'.join(g)} | {fp['lds'] / 1024:.1f} | {core:.1f} | {ho or '-'} | {fp['lds'] / 1024 - 32:+.1f} |")


def scenarios():
    """fused-group LDS under {current, lean} op implementations x C in {64, 32}, and the net HBM bytes including
    the chunk-size change of the formulation's scan <-> chunk crossing (kda_ir.crossing)."""
    global C
    groups = [["re_gate", "re_wy"], ["wy_bwd_a", "wy_bwd_b"], ["dav_a", "wy_bwd_a", "wy_bwd_b"],
              ["wy_bwd_b", "tail"], ["dav_a", "wy_bwd_a", "wy_bwd_b", "tail"]]
    print("core LDS per op (KB): current / lean, C=64 and C=32 (measured at C=64 in parentheses)")
    for op in ("re_gate", "re_wy", "dav", "wy_bwd_a", "wy_bwd_b", "tail"):
        print(f"  {op:9s} C=64 {core_lds(op, 64) / 1024:5.1f} / {core_lds(op, 64, True) / 1024:5.1f}  "
              f"C=32 {core_lds(op, 32) / 1024:5.1f} / {core_lds(op, 32, True) / 1024:5.1f}   (measured {CORE[op][0] / 1024:.1f})")
    x64 = ir.crossing(ir.formulation(), 64)
    cross = {c: sum((1 if d == "input->scan" else 2) * b for _, d, b in ir.crossing(ir.formulation(), c))
             for c in (64, 32)}
    segs, ops = segments("split")
    byname = {o.name: o for o in ops}
    base_k = [[o] for seg in segs for o in seg]
    print("\n| fused group | handoff saved B/tok*head | LDS KB (res): current C=64 | lean C=64 | current C=32 | lean C=32 |")
    print("|---|---:|---:|---:|---:|---:|")
    for g in groups:
        cells = []
        for c, lean in ((64, False), (64, True), (32, False), (32, True)):
            C = c
            CORE_MODE["lean"] = lean
            fp = kernel_footprint([byname[n] for n in g])
            cells.append(f"{fp['lds'] / 1024:.1f} ({fp['res'] if fp['lds'] <= LDS_CU else 0})")
        C = 64
        CORE_MODE["lean"] = False
        rest = [k for k in base_k if k[0].name not in g]
        hb0, _ = handoff_bytes(ops, base_k)
        hb1, _ = handoff_bytes(ops, rest + [[byname[n] for n in g]])
        print(f"| {'+'.join(g)} | {hb0 - hb1:.0f} | " + " | ".join(cells) + " |")
    print(f"\nformulation crossing (scan <-> chunk, W+R): C=64 {cross[64]:.0f}, C=32 {cross[32]:.0f} B/tok*head "
          f"-> going to C=32 costs {cross[32] - cross[64]:+.0f} B/tok*head before any fusion saving")


def main():
    global C
    ap = argparse.ArgumentParser()
    ap.add_argument("--dav", default="split", choices=("split", "whole"))
    ap.add_argument("--C", type=int, default=64, help="chunk size for the on-chip handoff sizes (op cores stay "
                                                       "the C=64 production footprints)")
    ap.add_argument("--budget", action="store_true")
    ap.add_argument("--scenarios", action="store_true")
    a = ap.parse_args()
    C = a.C
    if a.scenarios:
        scenarios()
        return
    if a.budget:
        budget([["re_gate", "re_wy"], ["re_l2norm", "re_gate", "re_wy"], ["wy_bwd_a", "wy_bwd_b"],
                ["dav_a", "wy_bwd_a", "wy_bwd_b"], ["wy_bwd_b", "tail"], ["dav_a", "wy_bwd_a", "wy_bwd_b", "tail"]])
        return
    segs, ops = segments(a.dav)
    print(f"segments (dav {a.dav}):")
    for i, s in enumerate(segs):
        print(f"  S{i}: {[o.name for o in s]}")
    prod = {"S0": [["re_l2norm"], ["re_gate"], ["re_wy"]] + ([["dav_v"]] if a.dav == "split" else []),
            "S1": [] if a.dav == "split" else [["dav"]],
            "S2": ([["dav_a"]] if a.dav == "split" else []) + [["wy_bwd_a"], ["wy_bwd_b"], ["tail"]]}
    per_seg = []
    for i, s in enumerate(segs):
        per_seg.append(list(compositions(s)) if s else [[]])
    rows = []
    for combo in itertools.product(*per_seg):
        kernels = [k for seg in combo for k in seg if k]
        fps = [kernel_footprint(k) for k in kernels]
        hb, det = handoff_bytes(ops, kernels)
        worst = ("infeasible" if any(f["verdict"] == "infeasible" for f in fps) else
                 "res1" if any(f["verdict"] == "res1" for f in fps) else "res2+")
        rows.append((hb, worst, fps, det))
    prod_b = None
    for hb, worst, fps, det in rows:
        if all(len(f["ops"]) == 1 for f in fps):
            prod_b = hb
    print(f"\nproduction-like grouping (every op its own kernel): handoff {prod_b:.0f} B/tok*head\n")
    print("| kernels (fused groups only) | handoff B/tok*head | vs production | worst verdict | per fused kernel: LDS KB / VGPR / res | history |")
    print("|---|---:|---:|---|---|---|")
    for hb, worst, fps, det in sorted(rows, key=lambda r: (r[1] == "infeasible", r[0])):
        fused = [f for f in fps if len(f["ops"]) > 1]
        if not fused:
            continue
        desc = "; ".join("+".join(f["ops"]) for f in fused)
        fp = "; ".join(f"{f['lds'] / 1024:.1f}/{f['vgpr']:.0f}/{f['res']}" for f in fused)
        hist = "; ".join(v for k, v in HISTORY.items() if any(all(x in f["ops"] for x in k) for f in fused))
        print(f"| {desc} | {hb:.0f} | {hb - prod_b:+.0f} | {worst} | {fp} | {hist} |")


if __name__ == "__main__":
    main()
