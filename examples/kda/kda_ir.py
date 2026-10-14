"""KDA forward + backward dataflow IR (phase 2, step 2 of docs/kda_phase1_summary.md).

Three layers, all quantities per token*head (B = bytes) so that every shape is T*H times the same numbers:
  1. LOGICAL  -- tensors and ops of the algorithm (FLA chunk_kda semantics, K = V = 128, chunk C), independent of
                 any kernel split.  Each op carries its parallel structure (row / chunk / scan_fwd / scan_bwd /
                 reduce_T), whether a scan is V-separable, MACs (triangular products counted at half), element-wise
                 lane-ops and transcendentals (kda_bound.py inventory, split per op).
  2. MATERIALIZED -- a logical tensor stored in HBM with a dtype and a layout (row-major, A / B / C-form native
                 fragments, bN state blocks, ...).  Several materializations of one logical tensor = layout copies.
  3. DECOMPOSITION -- kernels (launchers) = a set of ops + the materializations each reads from / writes to HBM.
                 PRODUCTION (P-54a, k3_g2.py default route) is the first instance; it is validated launcher by
                 launcher against the optrace interface records of the ceiling cards (16 shapes).
Formulation variants (W/U materialized vs W-free recurrence, keep-state vs recompute, dAqk in dav vs in s4b,
l2norm / gate materialized vs recomputed, chunk C) are declared on the logical ops for the step-4 search.

    python3 kda_ir.py validate      # production decomposition vs optrace, 16 shapes
    python3 kda_ir.py inventory     # logical work per token*head vs kda_bound.py
    python3 kda_ir.py interfaces    # production intermediate round trips per token*head
    python3 kda_ir.py crossing      # bytes that must cross between chunk-parallel and scan ops, per formulation
    python3 kda_ir.py breakdown     # production intermediate traffic by cause (crossing / copy / handoff / saved)
"""
import json
import os
import sys
from dataclasses import dataclass, field

K = V = 128
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CARDS = os.path.join(ROOT, "results", "kda-ceiling-cards-20261013", "cards.json")
DT = {"bf16": 2, "fp32": 4}


# ================================================================================================ logical
@dataclass
class Tensor:
    name: str
    shape: str          # row:X (X per token) | cc (C x C per chunk) | state (K x V per chunk) | vecK (K per chunk)
                        # | kc (K x C per chunk) | head:X (X per head, not per token)
    x: int = 0          # width for row / head shapes
    kind: str = "inter"  # input | output | saved | inter | param | grad_param
    note: str = ""

    def elems_per_tok(self, C):
        s = self.shape
        if s == "row":
            return self.x
        if s == "cc":
            return C                       # C*C per C tokens
        if s == "state":
            return K * V / C
        if s == "vecK":
            return K / C
        if s == "kc":
            return K                       # K*C per C tokens
        return 0.0                         # head:* tensors are per head, see elems_per_head

    def elems_per_head(self):
        return self.x if self.shape == "head" else 0


T_ = {}


def tensor(name, shape, x=0, kind="inter", note=""):
    T_[name] = Tensor(name, shape, x, kind, note)


# model inputs / outputs / parameters
tensor("q", "row", K, "input"); tensor("k", "row", K, "input"); tensor("v", "row", V, "input")
tensor("g_raw", "row", K, "input", "raw gate pre-activation (bf16)")
tensor("beta_raw", "row", 1, "input", "fp32 in the K3 contract")
tensor("A_log", "head", 1, "param"); tensor("dt_bias", "head", K, "param")
tensor("o", "row", V, "output"); tensor("do", "row", V, "input", "backward input")
tensor("dq_raw", "row", K, "output"); tensor("dk_raw", "row", K, "output"); tensor("dv", "row", V, "output")
tensor("dg_raw", "row", K, "output"); tensor("dbeta_raw", "row", 1, "output")
tensor("dA_log", "head", 1, "grad_param"); tensor("ddt_bias", "head", K, "grad_param")
# forward intermediates
tensor("rstd_q", "row", 1, "saved"); tensor("rstd_k", "row", 1, "saved")
tensor("qn", "row", K, note="l2-normalized q (bf16 RNE of q*rstd)"); tensor("kn", "row", K)
tensor("beta", "row", 1, note="sigmoid(beta_raw)*bscale")
tensor("g", "row", K, note="chunk-local cumsum of the log2 gate (fp32)")
tensor("Gn", "vecK", note="exp2(G_last) decays per chunk (P-52)")
tensor("Gn_raw", "vecK", note="G_last (deep-saturation recompute instance)")
tensor("Akk", "cc", note="strict-tril beta K_dec K_inv^T")
tensor("Tinv", "cc", note="(I + Akk)^-1 (FLA/production name: Akk after solve_tril)")
tensor("Aqk", "cc", note="tril scale Q_dec K_inv^T")
tensor("W", "row", K, note="T (beta K exp2(G))"); tensor("U", "row", V, note="T (beta V)")
tensor("Kg", "row", K, note="k exp2(G_last - G) (state-update operand)")
tensor("Qg", "row", K, note="q exp2(G) (inter-chunk output operand)")
tensor("kgB", "row", K, note="k exp2(G) in s4's vB layout (P3 construction)")
tensor("h", "state", note="state entering each chunk")
tensor("v_new", "row", V, note="U - W h")
# backward intermediates
tensor("dAqk", "cc", note="scale * tril(do v_new^T)")
tensor("dVn", "row", V, note="tril(Aqk)^T do (dv_local)")
tensor("dh", "state", note="state gradient entering each chunk (reverse)")
tensor("dv2", "row", V, note="dVn + Kg dh: dv_new total")
tensor("dA", "cc", note="s4a -> s4b: gradient w.r.t. Akk after the T^-1 chain")
tensor("dwT", "kc", note="s4a -> s4b: -dw transposed")
tensor("dq", "row", K, note="fp32 dq before the l2norm backward"); tensor("dk", "row", K)
tensor("dg", "row", K, note="fp32 dg before the reverse cumsum / gate backward"); tensor("db", "row", 1)


@dataclass
class Op:
    name: str
    phase: str           # fwd | bwd | recompute
    par: str             # row | chunk | scan_fwd | scan_bwd | reduce_T
    ins: tuple
    outs: tuple
    tri: float = 0.0     # triangular C x C x D products per chunk
    sq: float = 0.0      # square C x D x D products per chunk
    t3: float = 0.0      # triangular C x C inverse-type products per chunk
    valu: float = 0.0    # element-wise lane-ops per token
    sfu: float = 0.0     # transcendentals per token
    vsep: bool = True    # scan ops: separable over V slabs (no cross-slab reduction)
    variants: tuple = ()
    note: str = ""

    def mac_per_tok(self, C):
        return (self.tri * C * C * K / 2 + self.sq * C * K * V + self.t3 * C ** 3 / 6) / C


OPS = [
    # ------------------------------------------------------------------------------------- forward
    Op("l2norm_q", "fwd", "row", ("q",), ("qn", "rstd_q"), valu=2 * K, sfu=1, variants=("l2n_mat", "l2n_recompute")),
    # (per-op lane-op / transcendental split chosen so that the phase totals equal kda_bound.inventory("P1"))
    Op("l2norm_k", "fwd", "row", ("k",), ("kn", "rstd_k"), valu=2 * K, sfu=1),
    Op("beta_sig", "fwd", "row", ("beta_raw",), ("beta",), note="a few lane-ops per token, folded into the totals"),
    Op("gate_cumsum", "fwd", "chunk", ("g_raw", "A_log", "dt_bias"), ("g", "Gn", "Gn_raw"), valu=10 * K, sfu=2 * K,
       variants=("gate_mat", "gate_recompute")),
    Op("kkt_qkt", "fwd", "chunk", ("kn", "qn", "g", "beta"), ("Akk", "Aqk"), tri=2, valu=6 * K, sfu=3 * K,
       note="Akk, Aqk with the per-channel decays"),
    Op("tri_inverse", "fwd", "chunk", ("Akk",), ("Tinv",), t3=1, note="16x16 column sweep + block merge"),
    Op("wy_wu", "fwd", "chunk", ("Tinv", "kn", "v", "g", "beta"), ("W", "U", "Kg", "Qg"), tri=2,
       variants=("wu_mat", "wfree"), note="W = T(beta k e^G), U = T(beta v); Kg, Qg scaled operands"),
    Op("scan_fwd_o", "fwd", "scan_fwd", ("W", "U", "Kg", "Qg", "Aqk", "Gn"), ("o",), sq=3, tri=1, valu=4 * K + K * V / 64,
       note="v_new = U - W h; o = Qg h + Aqk v_new; h += Kg^T v_new (state never leaves the chip)"),
    # ------------------------------------------------------------------------------------- backward
    Op("re_l2norm", "bwd", "row", ("q", "k", "rstd_q", "rstd_k"), ("qn", "kn"), valu=2 * K),
    Op("re_gate", "bwd", "chunk", ("g_raw", "A_log", "dt_bias"), ("g", "Gn", "Gn_raw"), valu=10 * K, sfu=2 * K),
    Op("re_wy", "recompute", "chunk", ("Tinv", "kn", "qn", "v", "g", "beta"), ("W", "U", "Kg", "Qg", "kgB"), tri=2,
       note="production wu: W/U plus the scaled operands, several layouts"),
    Op("re_scan", "recompute", "scan_fwd", ("W", "U", "Kg", "Gn"), ("h", "v_new"), sq=2, valu=2 * V + K * V / 64,
       variants=("recompute", "keep_state"), note="keep_state: the forward writes h / v_new instead"),
    Op("dav", "bwd", "chunk", ("do", "v_new", "Aqk"), ("dAqk", "dVn"), tri=2, variants=("dAqk_dav", "dAqk_s4b")),
    Op("scan_dh", "bwd", "scan_bwd", ("Kg", "Qg", "W", "do", "dVn", "Gn"), ("dh", "dv2"), sq=3, valu=K * V / 64,
       note="dv2 = dVn + Kg dh; dh_prev = decay dh + Qg^T do - W^T dv2"),
    Op("wy_bwd_a", "bwd", "chunk", ("kgB", "v", "Tinv", "h", "dv2", "beta"), ("dv", "dwT", "dA", "db"), sq=1, tri=3,
       t3=2, valu=10 * K, note="s4a: dw = dv2 h^T, dA += dv2 v^T + dw kg^T, dv = T^T dv2 beta, dAkk = -T^T dA T^T"),
    Op("wy_bwd_b", "bwd", "chunk", ("qn", "kn", "g", "beta", "Tinv", "h", "dh", "do", "v_new", "dAqk", "dA", "dwT"),
           ("dq", "dk", "dg", "db"), sq=2, tri=5, valu=24 * K, sfu=3 * K,
       note="s4b: dq = do h^T, dk = v_new dh^T, dkgb = T^T(-dw), intra dq/dk from dAqk / dA tables"),
    Op("tail", "bwd", "chunk", ("dq", "dk", "dg", "db", "qn", "kn", "rstd_q", "rstd_k", "g_raw", "beta_raw",
                                "A_log", "dt_bias"), ("dq_raw", "dk_raw", "dg_raw", "dbeta_raw"), valu=18 * K, sfu=2 * K,
       note="reverse cumsum of dg, gate backward, l2norm backward, beta sigmoid backward"),
    Op("param_reduce", "bwd", "reduce_T", ("dg_raw",), ("dA_log", "ddt_bias"), valu=2 * K,
       note="fixed-order reduce over T (bwd_tail_reduce)"),
]
OPI = {o.name: o for o in OPS}


# ================================================================================================ materialized
@dataclass(frozen=True)
class Mat:
    tensor: str
    dtype: str = "bf16"
    layout: str = "row"   # row | natA | natB | natC | bN | aN | vB | rowT
    tag: str = ""         # distinguishes layout copies of one tensor (Wn / Wb, KgA / Kn ...)

    def bytes_per_tok(self, C):
        return T_[self.tensor].elems_per_tok(C) * DT[self.dtype]

    def bytes_per_head(self):
        return T_[self.tensor].elems_per_head() * DT[self.dtype]


def M(t, dtype="bf16", layout="row", tag=""):
    return Mat(t, dtype, layout, tag)


# ================================================================================================ decomposition
@dataclass
class Kernel:
    name: str
    ops: tuple
    reads: tuple
    writes: tuple
    internal: tuple = ()   # HBM handoffs between the sub-kernels of one launcher (not in the optrace)
    note: str = ""

    def io_bytes(self, C, H):
        rd = sum(m.bytes_per_tok(C) for m in self.reads)
        wr = sum(m.bytes_per_tok(C) for m in self.writes)
        rdh = sum(m.bytes_per_head() for m in self.reads)
        wrh = sum(m.bytes_per_head() for m in self.writes)
        return rd, wr, rdh, wrh


# production layouts (P-54a): names follow k3_g2.py
Wn, Wb = M("W", layout="natA", tag="Wn"), M("W", layout="natB", tag="Wb")
Un = M("U", layout="natC", tag="Un")
KgA, Kn = M("Kg", layout="natA", tag="KgA"), M("Kg", layout="natB", tag="Kn")
Qn, QgA = M("Qg", layout="natB", tag="Qn"), M("Qg", layout="natA", tag="QgA")
kgB = M("kgB", layout="vB")
Gn = M("Gn", "fp32", "vecK"); Gn_raw = M("Gn_raw", "fp32", "vecK")
AqkA, Aqk_row = M("Aqk", layout="natA", tag="AqkA"), M("Aqk", layout="row", tag="Aqk")
Tinv_row = M("Tinv", layout="row", tag="Akk")
bN_h, Vn = M("h", layout="bN"), M("v_new", layout="natC", tag="Vn")
dAqk_row = M("dAqk", "fp32", "row"); dVn = M("dVn", layout="natC"); dOT = M("do", layout="natB", tag="dOT")
bN_dh, aN_dv2 = M("dh", layout="bN"), M("dv2", layout="aN")
dA_row, dwT = M("dA", "fp32", "row"), M("dwT", layout="natB")
q, k, v, g_raw, do = M("q"), M("k"), M("v"), M("g_raw"), M("do")
beta_raw, beta = M("beta_raw", "fp32"), M("beta", "fp32")
A_log, dt_bias = M("A_log", "fp32"), M("dt_bias", "fp32")
rq, rk = M("rstd_q", "fp32"), M("rstd_k", "fp32")
qn, kn = M("qn"), M("kn")
g32 = M("g", "fp32")
dq32, dk32, dg32, db32 = M("dq", "fp32"), M("dk", "fp32"), M("dg", "fp32"), M("db", "fp32")

PRODUCTION = [
    # forward (KDA_G2_FI=4 single-pass prep; beta = head_beta_sigmoid is a separate tiny launch, not traced)
    Kernel("fwd_prep", ("l2norm_q", "l2norm_k", "gate_cumsum", "kkt_qkt", "tri_inverse", "wy_wu"),
           (q, k, v, g_raw, A_log, dt_bias, beta),
           (Wn, Kn, Un, QgA, AqkA, Aqk_row, Tinv_row, rq, rk, Gn)),
    Kernel("fwd_h_o", ("scan_fwd_o",), (Wn, Kn, Un, Gn, QgA, AqkA), (M("o"),)),
    # backward (recompute policy, S4 split route, P-17 / P-20 / P-22 copies removed)
    Kernel("l2n_apply", ("re_l2norm",), (q, k, rq, rk), (qn, kn)),
    Kernel("head_gate", ("re_gate",), (g_raw, A_log, dt_bias), (g32,),
           note="Gn / Gn_raw come from to_gn + exp2_ (small torch / HIP ops, not in the optrace)"),
    Kernel("wu", ("re_wy",), (kn, v, qn, beta, Tinv_row, g32), (Wn, Wb, Un, KgA, Kn, Qn, kgB)),
    Kernel("fwd_h_bn", ("re_scan",), (Wn, Kn, Un, Gn, Gn_raw), (bN_h, Vn)),
    Kernel("dav", ("dav",), (do, Vn, Aqk_row), (dAqk_row, dVn, dOT)),
    Kernel("dhu_bn", ("scan_dh",), (KgA, Qn, Wb, dOT, dVn, Gn), (bN_dh, aN_dv2)),
    Kernel("wy_s4_split", ("wy_bwd_a", "wy_bwd_b"),
           (kgB, kn, g32, beta, Tinv_row, bN_h, bN_dh, aN_dv2, qn, v, dAqk_row, do, Vn),
           (dq32, dk32, M("dv"), dg32, db32),
           internal=(dA_row, dwT), note="s4a writes dA (fp32) + dwT, s4b reads them"),
    Kernel("bwd_tail", ("tail", "param_reduce"),
           (dq32, dk32, dg32, db32, qn, kn, rq, rk, g_raw, A_log, dt_bias, beta_raw),
           (M("dq_raw"), M("dk_raw"), M("dg_raw"), M("dA_log", "fp32"), M("ddt_bias", "fp32"), M("dbeta_raw", "fp32"))),
]


# ================================================================================================ formulations
# Alternative op sets for the step-4 search.  Each formulation = the forward ops + the backward ops; the
# phase-crossing analysis below gives, per formulation, the tensors that MUST cross between chunk-parallel ops
# and scan ops (any decomposition that alternates the two parallel structures pays at least these bytes).
VARIANT_OPS = {
    # W-free (FROST-style): the scans take k, v, T^-1 and the gate instead of the materialized W / U
    "kq_scale": Op("kq_scale", "fwd", "chunk", ("kn", "qn", "g"), ("Kg", "Qg"), note="scaled operands only"),
    "scan_fwd_o_wfree": Op("scan_fwd_o_wfree", "fwd", "scan_fwd", ("kn", "v", "Tinv", "beta", "g", "Kg", "Qg", "Aqk", "Gn"),
                           ("o",), sq=3, tri=2, valu=6 * K + K * V / 64,
                           note="Y = beta (V - K_dec h); v_new = T^-1 Y: one more dependent C x C x V product per step; "
                                "K_dec = k exp2(G) is formed inside the scan (needs g)"),
    "re_kq_scale": Op("re_kq_scale", "recompute", "chunk", ("kn", "qn", "g"), ("Kg", "Qg", "kgB")),
    "re_scan_wfree": Op("re_scan_wfree", "recompute", "scan_fwd", ("kn", "v", "Tinv", "beta", "g", "Kg", "Gn"),
                        ("h", "v_new"), sq=2, tri=1, valu=2 * K + K * V / 64),
    "scan_dh_wfree": Op("scan_dh_wfree", "bwd", "scan_bwd", ("Kg", "Qg", "kn", "Tinv", "beta", "g", "do", "dVn", "Gn"),
                        ("dh", "dv2"), sq=3, tri=1, valu=2 * K + K * V / 64,
                        note="W^T dv2 = K_dec^T beta (T^T dv2): one more dependent product per step"),
}
REPLACE_WFREE = {"wy_wu": "kq_scale", "scan_fwd_o": "scan_fwd_o_wfree", "re_wy": "re_kq_scale",
                 "re_scan": "re_scan_wfree", "scan_dh": "scan_dh_wfree"}


def _with(o, **kw):
    d = dict(name=o.name, phase=o.phase, par=o.par, ins=o.ins, outs=o.outs, tri=o.tri, sq=o.sq, t3=o.t3,
             valu=o.valu, sfu=o.sfu, vsep=o.vsep, variants=o.variants, note=o.note)
    d.update(kw)
    return Op(**d)


def formulation(wy="wu_mat", state="recompute", dAqk="dAqk_dav", C=64):
    """op list (in execution order) of one formulation:
      wy    : wu_mat (production: W / U materialized by a chunk op) | wfree (scans consume k, v, T^-1, g)
      state : recompute (production: the backward re-runs the forward scan) | keep_state (the forward emits h, v_new)
      dAqk  : dAqk_dav (production) | dAqk_s4b (P-55: recomputed by the s4b op)"""
    ops = []
    for o in OPS:
        if wy == "wfree" and o.name in REPLACE_WFREE:
            o = VARIANT_OPS[REPLACE_WFREE[o.name]]
        if state == "keep_state":
            if o.name.startswith("re_scan"):
                continue
            if o.name.startswith("scan_fwd_o"):
                o = _with(o, outs=o.outs + ("h", "v_new"), note=o.note + " | keep-state: also writes h, v_new")
        if dAqk == "dAqk_s4b":
            if o.name == "dav":
                o = _with(o, outs=tuple(t for t in o.outs if t != "dAqk"), tri=o.tri - 1)
            elif o.name == "wy_bwd_b":
                o = _with(o, ins=tuple(t for t in o.ins if t != "dAqk") + ("v_new",), tri=o.tri + 1)
        ops.append(o)
    return ops


MIN_DT = {"g": "fp32", "dq": "fp32", "dk": "fp32", "dg": "fp32", "db": "fp32", "dA": "fp32", "dAqk": "fp32",
          "beta": "fp32", "Gn": "fp32", "Gn_raw": "fp32", "rstd_q": "fp32", "rstd_k": "fp32"}


def crossing(ops, C=64):
    """Tensors that cross between scan ops and chunk / row ops, versioned like the optrace (a tensor recomputed in
    the backward is a new version).  Each crossing costs one write + one read in its minimal production dtype
    (consumers on the same side are assumed fused); model inputs read directly by a scan count one read."""
    scan = lambda o: o.par.startswith("scan")
    ver, n_ver, seen, out = {}, {}, set(), []
    for o in ops:
        for t in o.ins:
            if t in ver:
                po, vid = ver[t]
                if scan(po) != scan(o) and (t, vid) not in seen:
                    seen.add((t, vid))
                    out.append((f"{t}@{po.name}", "scan->chunk" if scan(po) else "chunk->scan",
                                T_[t].elems_per_tok(C) * DT[MIN_DT.get(t, "bf16")]))
            elif scan(o) and T_[t].kind == "input" and (t, "in") not in seen:
                seen.add((t, "in"))
                out.append((t, "input->scan", T_[t].elems_per_tok(C) * DT[MIN_DT.get(t, "bf16")]))
        for t in o.outs:
            n_ver[t] = n_ver.get(t, 0) + 1
            ver[t] = (o, n_ver[t])
    return out


def chain_steps(ops):
    """sequential scan passes and the dependent products per step (critical-path proxy for the scans)"""
    return [(o.name, o.sq + o.tri) for o in ops if o.par.startswith("scan")]


# ================================================================================================ production breakdown
# Why production moves 18.6 KB of intermediates per token*head when the formulation only forces ~8.2 KB across the
# chunk / scan boundary.  Every production materialization is assigned one cause:
#   crossing  : the first materialization of a tensor version that must cross between chunk and scan ops
#   copy      : a second layout of the same version (one consumer wants another fragment form)
#   extra_read: further reads of a crossing tensor by additional kernels on the same side
#   handoff   : chunk op -> chunk op in a different kernel (the chunk-parallel backward split into several kernels
#               because one CTA cannot hold the working set: 64 KB LDS / 256 VGPR)
#   saved     : forward -> backward residency of the memory policy (Aqk, T^-1, rstd, beta)
PROD_CAUSE = {
    "Wn": "crossing", "Un": "crossing", "Kn": "crossing", "QgA": "crossing", "AqkA": "crossing", "Gn": "crossing",
    "Gn_raw": "crossing", "h": "crossing", "Vn": "crossing", "dVn": "crossing", "dh": "crossing", "dv2": "crossing",
    "KgA": "crossing", "Qn": "crossing",
    "Wb": "copy", "dOT": "copy",
    "Aqk": "saved", "Akk": "saved", "rstd_q": "saved", "rstd_k": "saved", "beta": "saved",
    "qn": "handoff", "kn": "handoff", "g": "handoff", "kgB": "handoff", "dAqk": "handoff", "dA": "handoff",
    "dwT": "handoff", "dq": "handoff", "dk": "handoff", "dg": "handoff", "db": "handoff",
}


def breakdown(decomp=PRODUCTION, C=64):
    wr, rd = interfaces(decomp, C)
    cause = {}
    for m in set(wr) | set(rd):
        key = m.tag or m.tensor
        b = m.bytes_per_tok(C)
        c = PROD_CAUSE.get(key, "?")
        w, r = wr.get(m, 0), rd.get(m, 0)
        if c == "crossing":
            # one write + one read is the crossing; the rest are extra reads (the forward and the backward
            # versions of Wn / Un / Kn are counted as two crossings each: write 2, read 2)
            n = min(w, r)
            cause["crossing"] = cause.get("crossing", 0) + 2 * n * b
            cause["extra_read"] = cause.get("extra_read", 0) + (w + r - 2 * n) * b
        else:
            cause[c] = cause.get(c, 0) + (w + r) * b
    return cause


# ================================================================================================ checks
def validate(decomp=PRODUCTION, C=64):
    cards = json.load(open(CARDS))
    worst, rows = 0.0, []
    for sh, d in cards.items():
        T, H = d["T"], d["H"]
        L = d["launchers"]
        for kern in decomp:
            if kern.name not in L:
                rows.append((sh, kern.name, "missing in optrace"))
                continue
            rd, wr, rdh, wrh = kern.io_bytes(C, H)
            pin = rd * T * H + rdh * H
            pout = wr * T * H + wrh * H
            min_, mout = L[kern.name]["in_bytes"], L[kern.name]["out_bytes"]
            ein, eout = pin / min_ - 1, pout / mout - 1
            worst = max(worst, abs(ein), abs(eout))
            rows.append((sh, kern.name, f"in {pin / 2**20:8.1f} / {min_ / 2**20:8.1f} MB ({100 * ein:+.3f}%)  "
                                        f"out {pout / 2**20:8.1f} / {mout / 2**20:8.1f} MB ({100 * eout:+.3f}%)"))
    return worst, rows


def inventory(C=64):
    tot = {"fwd": [0, 0, 0], "bwd": [0, 0, 0], "recompute": [0, 0, 0]}
    for o in OPS:
        t = tot[o.phase]
        t[0] += o.mac_per_tok(C)
        t[1] += o.valu
        t[2] += o.sfu
    return tot


def interfaces(decomp=PRODUCTION, C=64):
    """per token*head: bytes of every intermediate materialization written and read inside the operator."""
    ext = {"input", "output", "param", "grad_param"}
    inter = lambda m: T_[m.tensor].kind not in ext or m.tag or m.layout != "row"   # layout copies of inputs count
    wr, rd = {}, {}
    for kern in decomp:
        for m in kern.writes + kern.internal:
            if inter(m):
                wr[m] = wr.get(m, 0) + 1
        for m in kern.reads + kern.internal:
            if inter(m):
                rd[m] = rd.get(m, 0) + 1
    return wr, rd


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if cmd == "validate":
        worst, rows = validate()
        for r in rows:
            if r[0] in ("16384_12", "65536_96"):
                print(f"{r[0]:9s} {r[1]:12s} {r[2]}")
        print(f"\n16 shapes x {len(PRODUCTION)} launchers: worst |relative error| = {100 * worst:.4f}%")
    elif cmd == "inventory":
        sys.path.insert(0, os.path.dirname(__file__))
        import kda_bound as kb
        mac, valu, sfu, *_ = kb.inventory("P1")
        inv = inventory()
        print("| phase | IR MAC/tok | kda_bound MAC/tok | IR lane-ops | kda_bound | IR transc. | kda_bound |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for ph in ("fwd", "bwd", "recompute"):
            print(f"| {ph} | {inv[ph][0]:.0f} | {mac[ph]:.0f} | {inv[ph][1]:.0f} | {valu[ph]:.0f} | {inv[ph][2]:.0f} | {sfu[ph]:.0f} |")
    elif cmd == "crossing":
        print("| formulation | chunk -> scan B/tok*head | scan -> chunk | inputs read by scans | total (W+R) | scans | tensors |")
        print("|---|---:|---:|---:|---:|---|---|")
        for wy in ("wu_mat", "wfree"):
            for st in ("recompute", "keep_state"):
                for C in (32, 64, 128):
                    x = crossing(formulation(wy, st, C=C), C)
                    cs = sum(b for _, d, b in x if d == "chunk->scan")
                    sc = sum(b for _, d, b in x if d == "scan->chunk")
                    ins = sum(b for _, d, b in x if d == "input->scan")
                    names = ", ".join(f"{t}" for t, d, b in x)
                    ch = chain_steps(formulation(wy, st, C=C))
                    print(f"| {wy} / {st} / C={C} | {cs:.0f} | {sc:.0f} | {ins:.0f} | {2 * (cs + sc) + ins:.0f} | "
                          f"{len(ch)} passes, dependent products/step {[p for _, p in ch]} | {names} |")
    elif cmd == "breakdown":
        cz = breakdown()
        tot = sum(cz.values())
        x = crossing(formulation(), 64)
        minx = sum(2 * b for _, d, b in x if d != "input->scan") + sum(b for _, d, b in x if d == "input->scan")
        print("| cause | B/tok*head (W+R) | share |")
        print("|---|---:|---:|")
        for c in ("crossing", "extra_read", "copy", "handoff", "saved"):
            print(f"| {c} | {cz.get(c, 0):.0f} | {100 * cz.get(c, 0) / tot:.0f}% |")
        print(f"| total | {tot:.0f} | |")
        print(f"\nformulation-forced crossing (kda_ir crossing, decomposition-independent): {minx:.0f} B/tok*head")
    elif cmd == "interfaces":
        wr, rd = interfaces()
        tot_w = tot_r = 0
        print("| intermediate | dtype | layout | B/tok*head | writes | reads |")
        print("|---|---|---|---:|---:|---:|")
        for m in sorted(set(wr) | set(rd), key=lambda m: (m.tensor, m.tag)):
            b = m.bytes_per_tok(64)
            tot_w += b * wr.get(m, 0)
            tot_r += b * rd.get(m, 0)
            print(f"| {m.tag or m.tensor} | {m.dtype} | {m.layout} | {b:.0f} | {wr.get(m, 0)} | {rd.get(m, 0)} |")
        print(f"\nintermediate traffic per token*head: write {tot_w:.0f} B, read {tot_r:.0f} B")


if __name__ == "__main__":
    main()
