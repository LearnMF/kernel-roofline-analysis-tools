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
            parts = [(k.name, self._kernel(k, T, H)) for k in dec
                     if phase == "fwd+bwd" or (k.name in ("fwd_prep", "fwd_h_o")) == (phase == "fwd")]
            if parts:
                w.splits[name] = parts
        w.notes.append("T_split p82 is the production route only for H <= 20 (KDA_G2_SAVE_H_MAX_H); "
                       "above it production is p77b")
        return w

    @staticmethod
    def _kernel(k, T: int, H: int) -> Work:
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
        return w


OPS = (Kda(),)
