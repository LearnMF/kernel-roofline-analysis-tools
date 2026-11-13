"""KDA (chunked gated delta rule with per-channel decay, FLA chunk_kda semantics), d_k = d_v = 128.

T_theory: the fused operator (examples/kda/kda_bound.py): irreducible HBM bytes, algorithmic MMAC
FLOPs, vector lane-ops and transcendentals, and the chunk recurrence -- nt = T / C dependent steps
per sequential pass (forward with o, then the state-only passes of the backward), whose per-step
latency is a machine property.

T_split: the same work cut into the kernels of a decomposition (kra.ops.kda_ir): every kernel reads
and writes its interface tensors through HBM, so each kernel is bounded on its own and the bounds
add.  O_handoff = T_split / T_theory is what that kernel split costs before any implementation
detail.  Decompositions: p54a (10-13 production, validated byte-exact against optrace), p77b
(forward saves qn/kn), p82 (also saves h; production only for H <= 20).

shape: T (tokens per sequence x batch), H (heads); optional save_h=1 for the keep-state policy (P2)
in T_theory.
"""
from __future__ import annotations

from . import Op, Work
from . import kda_ir as ir

C, D = 64, 128
SCAN_CTAS_PER_HEAD = D // 32          # one CTA per (head, 32-column V slab)
VALU_PACKED = 0.75                    # half the lane-ops are packable fp32 pairs (2 per slot)


def _required_bytes(save_h: bool) -> dict:
    """kda_bound.inventory: per token*head, inputs read once, outputs written once, saved tensors."""
    saved = 8 + 2 * C + 2 * C                       # rstd q/k, Aqk, T^-1 (bf16 C x C per C tokens)
    rb = {"fwd": 4 * 2 * D + 4, "bwd": 4 * 2 * D + 4 + 2 * D + saved}
    wb = {"fwd": 2 * D + saved, "bwd": 4 * 2 * D + 4}
    if save_h:
        wb["fwd"] += 2 * D * D / C
        rb["bwd"] += 2 * D * D / C
    return {"rd": rb, "wr": wb}


def _ops_work(ops, N: float) -> tuple:
    mac = sum(o.mac_per_tok(C) for o in ops) * N
    return 2.0 * mac, sum(o.valu for o in ops) * N, sum(o.sfu for o in ops) * N


def _step_key(op) -> str:
    return "kda_step_with_o" if op.name == "scan_fwd_o" else "kda_step_state"


class Kda(Op):
    name = "kda"
    params = ("T", "H")
    phases = ("fwd", "bwd", "fwd+bwd")

    def work(self, shape: dict, dtype: str, phase: str) -> Work:
        self.check(shape, phase)
        if dtype != "bf16":
            raise SystemExit("kda plugin models the bf16 operator only")
        T, H = int(shape["T"]), int(shape["H"])
        if T % C:
            raise SystemExit("kda: T must be a multiple of the chunk size %d" % C)
        save_h = bool(shape.get("save_h", 0))
        N, nt = float(T * H), T // C
        want = {"fwd": ("fwd",), "bwd": ("bwd", "recompute"), "fwd+bwd": ("fwd", "bwd", "recompute")}[phase]
        ops = [o for o in ir.OPS if o.phase in want and not (save_h and o.name == "re_scan")]
        flops, lanes, sfu = _ops_work(ops, N)
        rq = _required_bytes(save_h)
        ph = [p for p in ("fwd", "bwd") if p in want]
        w = Work(flops=flops, bytes_rd=sum(rq["rd"][p] for p in ph) * N,
                 bytes_wr=sum(rq["wr"][p] for p in ph) * N, compute_eff="mmac_tile_eff", vmem_issue=True,
                 other={"valu_lane": VALU_PACKED * lanes, "sfu_exp2": sfu},
                 serial=[(nt, _step_key(o), 1) for o in ops if o.par.startswith("scan")])
        w.bytes_min = w.bytes_rd + w.bytes_wr
        w.working_set = float(2 * D * D * 4)       # one head's fp32 state + its gradient must stay on chip
        w.notes.append("T_theory: fused operator, policy %s (%s); recurrence %d passes x %d steps"
                       % ("P2" if save_h else "P1", "keep state" if save_h else "backward recomputes the state",
                          len(w.serial), nt))
        for name, dec in ir.DECOMPOSITIONS.items():
            parts = [(k.name, self._kernel(k, T, H, phase)) for k in dec
                     if phase == "fwd+bwd" or (k.name in ("fwd_prep", "fwd_h_o")) == (phase == "fwd")]
            if parts:
                w.splits[name] = parts
        w.notes.append("T_split p82 is the production route only for H <= 20 (KDA_G2_SAVE_H_MAX_H); "
                       "above it production is p77b")
        return w

    @staticmethod
    def _kernel(k, T: int, H: int, phase: str = "fwd+bwd") -> Work:
        rd, wr, rdh, wrh = k.io_bytes(C, H)
        N = float(T * H)
        ops = [ir.OPI[o] for o in k.ops]
        flops, lanes, sfu = _ops_work(ops, N)
        w = Work(flops=flops, bytes_rd=rd * N + rdh * H, bytes_wr=wr * N + wrh * H,
                 compute_eff="mmac_tile_eff", vmem_issue=True,
                 other={"valu_lane": VALU_PACKED * lanes, "sfu_exp2": sfu},
                 serial=[(T // C, _step_key(o), H * SCAN_CTAS_PER_HEAD) for o in ops if o.par.startswith("scan")])
        if k.name == "fwd_h_o" and any(m.tensor == "h" for m in k.writes):
            w.serial = [(T // C, "kda_step_with_o", H * SCAN_CTAS_PER_HEAD)]
        w.bytes_min = w.bytes_rd + w.bytes_wr
        w.occupancy = _occupancy_of(k.ops, T, H, phase)
        return w


    # --- capacity split search entry point (see capacity_graph below) -------------------------
    def capacity_graphs(self, shape: dict, phase: str, residency_mode: str = "shape",
                        vgpr_mode: str = "full") -> list:
        """[(label, CapGraph)]: one graph per formulation variant the operator admits.  The search
        then spans formulation x kernel grouping, so a better formulation can win on a machine whose
        capacity or bandwidth makes the production one the wrong choice."""
        self.check(shape, phase)
        return [("/".join("%s=%s" % kv for kv in sorted(f.items())),
                 capacity_graph(shape, phase, residency_mode, f, vgpr_mode)) for f in FORMULATIONS]


OPS = (Kda(),)


# --------------------------------------------------------------------------------- capacity graph (M2b)
# The split search needs the operator's dataflow plus, for one tile of work (one (chunk, head) tile:
# C tokens x D channels, 256 threads), how much of each tensor has to be on chip.  Both are derived from
# the IR, NOT from the production kernels' measured LDS/VGPR: a measured footprint is the footprint of
# today's implementation on this card, so searching with it would be fitting the answer.  The rules:
#
#   core LDS  -- an op that contracts over the token axis (tri / sq / t3 products) has to make one
#                operand tile visible to all lanes, so the largest input tile is staged in LDS; a
#                triangular inverse (t3) also keeps its C x C matrix there across the sweep.
#                Ops that only touch their own lane's rows (valu / sfu) need none.
#   core regs -- the accumulators (output tiles, fp32 for products) plus the operands not staged in LDS
#                (see _core_regs), plus a fixed per-thread addressing allowance per kernel.
#   handoffs  -- a tensor produced and consumed inside one kernel stays live across the ops between
#                them: a per-token row tile keeps the lane mapping (registers), a C x C / state / K x C
#                tile has to go through LDS.
#
# Residency mode "lds" charges every handoff to LDS (upper bound on the footprint, prunes more); mode
# "shape" applies the rule above.  Both are reported so the sensitivity is visible.
from ..model.capacity import CapGraph, CapOp, TileTensor  # noqa: E402

THREADS = 256
LANE_SHAPES = ("row",)          # per-token tiles: the lane mapping carries them in registers


def _tile_bytes(name: str) -> float:
    t = ir.T_[name]
    dt = ir.DT[ir.MIN_DT.get(name, "bf16")]
    if t.shape == "head":
        return t.elems_per_head() * dt
    return t.elems_per_tok(C) * C * dt


def _hbm_bytes(name: str, T: int, H: int) -> float:
    t = ir.T_[name]
    dt = ir.DT[ir.MIN_DT.get(name, "bf16")]
    if t.shape == "head":
        return t.elems_per_head() * dt * H
    return t.elems_per_tok(C) * dt * T * H


def _core_lds(op) -> float:
    if not (op.tri or op.sq or op.t3):
        return 0.0
    staged = max((_tile_bytes(t) for t in op.ins), default=0.0)
    if op.t3:
        staged += max((_tile_bytes(t) for t in op.outs), default=0.0)
    return staged


def _is_product(op) -> bool:
    return bool(op.tri or op.sq or op.t3)


def _staged(op) -> str:
    """The input tile a product op stages in LDS (its largest), '' for lane-local ops."""
    if not _is_product(op):
        return ""
    return max(op.ins, key=_tile_bytes, default="")


def _core_regs(op) -> float:
    """Per-CTA accumulator bytes: the output tiles, fp32 for anything a product accumulates."""
    acc = 0.0
    for t in op.outs:
        b = _tile_bytes(t)
        if _is_product(op):
            b *= 4.0 / ir.DT[ir.MIN_DT.get(t, "bf16")]
        acc += b
    return acc


def _operands(op) -> tuple:
    """Operand tiles a product op holds while it runs, besides the one it stages in LDS: each one sits in
    registers or, when registers run out, in LDS (the solver decides per machine).  The op's products run one
    after another and one product has two operands, so at most the two largest non-staged inputs are live
    at once (the IR does not say which inputs pair up; the two largest is the safe side).  Lane-local ops
    stream their inputs element by element, so they hold none for the whole tile."""
    if not _is_product(op):
        return ()
    st = _staged(op)
    tiles = sorted((_tile_bytes(t) for t in op.ins if t != st and ir.T_[t].shape != "head"), reverse=True)
    return tuple(tiles[:2])


# Addressing, loop counters, lane / chunk indices: per-thread registers every kernel holds whatever its
# dataflow.  An assumption about the compiler, not a measurement: reported in the ledger and the result
# is shown with and without it.
REG_BASE_VGPR = 16


def _ctas(op, nt: int, H: int) -> int:
    """Tiles of work the op's parallel structure admits: a chunk / row op has one per (chunk, head), a
    scan is sequential over chunks so only its heads and separable V slabs are parallel, a reduction over
    the sequence has one per head.  This is the operator's parallelism, not a launch configuration."""
    if op.par.startswith("scan"):
        return H * (SCAN_CTAS_PER_HEAD if op.vsep else 1)
    if op.par == "reduce_T":
        return nt * H          # a reduction over the sequence is tree-parallel: partials then a tiny pass
    return nt * H


def _occupancy_of(op_names, T: int, H: int, phase: str) -> dict | None:
    """Grid and on-chip footprint of a hand-written decomposition's kernel, derived the same way as for
    a searched grouping, so the two are priced with the same terms.  None when no formulation of the IR
    contains all of the kernel's ops (then the concurrency terms are left out instead of guessed)."""
    from ..model.capacity import footprint, occupancy as _occ
    key = (T, H, phase)
    if _GRAPH_CACHE.get("key") != key:
        _GRAPH_CACHE.clear()
        _GRAPH_CACHE["key"] = key
        _GRAPH_CACHE["graphs"] = [capacity_graph({"T": T, "H": H}, phase, formulation=f) for f in FORMULATIONS]
    want = set(op_names)
    for g in _GRAPH_CACHE["graphs"]:
        have = {o.name: o for o in g.ops}
        if want <= set(have):
            kern = [have[n] for n in op_names]
            return _occ(g, kern, footprint(g, kern))
    return None


_GRAPH_CACHE: dict = {}


def capacity_graph(shape: dict, phase: str = "fwd+bwd", residency_mode: str = "shape",
                   formulation: dict | None = None, vgpr_mode: str = "full") -> CapGraph:
    T, H = int(shape["T"]), int(shape["H"])
    if T % C:
        raise SystemExit("kda: T must be a multiple of the chunk size %d" % C)
    N, nt = float(T * H), T // C
    want = {"fwd": ("fwd",), "bwd": ("bwd", "recompute"), "fwd+bwd": ("fwd", "bwd", "recompute")}[phase]
    form = dict(wy="wu_mat", state="recompute", dAqk="dAqk_dav")
    form.update(formulation or {})
    ops = [o for o in ir.formulation(**form) if o.phase in want]
    names = {o.name for o in ops}

    tensors = {}
    for o in ops:
        for t in tuple(o.ins) + tuple(o.outs):
            if t in tensors:
                continue
            kind = ir.T_[t].kind
            kind = "input" if kind in ("input", "param", "saved") else \
                   "output" if kind in ("output", "grad_param") else "inter"
            res = ""
            if residency_mode == "shape" and ir.T_[t].shape in LANE_SHAPES:
                res = "reg"
            tensors[t] = TileTensor(t, _tile_bytes(t), _hbm_bytes(t, T, H), kind, res,
                                    ir.MIN_DT.get(t, "bf16"))
    cops = []
    for o in ops:
        flops, lanes, sfu = _ops_work([o], N)
        cops.append(CapOp(
            name=o.name, ins=tuple(t for t in o.ins if t in tensors),
            outs=tuple(t for t in o.outs if t in tensors),
            barrier=o.par.startswith("scan") or o.par == "reduce_T",
            core_lds=_core_lds(o), core_reg_bytes=_core_regs(o),
            movable=_operands(o) if vgpr_mode == "full" else (), flops=flops,
            other={"valu_lane": VALU_PACKED * lanes, "sfu_exp2": sfu},
            serial=[(nt, _step_key(o), H * SCAN_CTAS_PER_HEAD)] if o.par.startswith("scan") else [],
            ctas=_ctas(o, nt, H)))
    return CapGraph(ops=cops, tensors=tensors, threads=THREADS, tiles=nt * H,
                    reg_base_vgpr=REG_BASE_VGPR if vgpr_mode == "full" else 0, compute_eff="mmac_tile_eff",
                    vmem_issue=True,
                    note="KDA %s, formulation %s, chunk C=%d, tile = one (chunk, head) on %d threads, "
                         "residency rule %r, vgpr rule %r; footprints derived from the IR, not from measured kernels "
                         "(%d ops, %d tensors)"
                         % (phase, "/".join("%s=%s" % kv for kv in sorted(form.items())), C, THREADS,
                            residency_mode, vgpr_mode, len(cops), len(tensors)))


FORMULATIONS = ({"state": "recompute"}, {"state": "keep_state"},
                {"state": "recompute", "wy": "wfree"}, {"state": "keep_state", "wy": "wfree"},
                {"state": "recompute", "dAqk": "dAqk_s4b"}, {"state": "keep_state", "dAqk": "dAqk_s4b"})
