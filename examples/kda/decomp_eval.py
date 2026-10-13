"""Decomposition evaluator (structural search, step 1): predicts the time of a CHANGED kernel split from the
measured ceiling cards (results/kda-ceiling-cards-20261013/cards.json: 16 shapes x every production kernel:
time, CTAs, residency, rounds, LDS/VGPR, per-resource stream floors, VMEM/VALU/LDS instruction counts) and is
back-tested against the fusion / recompute / geometry experiments we already ran before it is trusted.

Kernel time model (exact for every measured kernel by construction):
    t = max( R * lam ,  F * imb * e )
    R   = rounds = ceil(CTAs / (80 * resident)),   resident from LDS (64 KB / CU) and VGPR (R38: 768 / VGPR waves
          per SIMD, i.e. 3 at 256) and the 40-wave CU limit
    lam = per-CTA latency: t / R when R >= 2; for single-round kernels whose CTAs share CUs (recurrences at
          H >= 24) the ISOLATED latency, i.e. the same kernel at H = 12 (48 CTAs, <= 1 per CU)
                                                    -- the latency / issue term (B4b, P-13, P-51 live here)
    F   = max per-resource stream floor (HBM bytes, TA, VALU, LDS, MMAC, chain; cards f_*)
    imb = ceil(CTAs/80) / (CTAs/80): the busiest CU sets the time (P-49: 96 wide CTAs on 80 CUs)
    e   = calibrated so that the measured kernel is reproduced -- the throughput term's efficiency
Transformations (each returns a new Kernel, then t is recomputed):
    merge(host, guest)   same grid: lam adds (one CTA now runs both instruction streams), floors add minus the
                         removed traffic, LDS = host + extra (reuse given explicitly), VGPR = max + extra
    add_work(k, ...)     consumer recompute / extra emits: lam += instructions x cycles (VALU 8, transcendental 16,
                         VMEM: the kernel's own measured cycles per instruction), floors += the added demand
    remove_bytes(k, ...) dropped loads / stores: floors -= demand, lam -= beta x the traffic's share of lam
    geometry(k, ...)     CTA count / per-CTA work change (wide slabs): lam scales with the per-CTA work
Out of scope (flagged, not modelled): L2-capacity effects (P-49 64K), stream-level concurrency (F9).
    python3 decomp_eval.py backtest | regimes
"""
import json
import math
import os
import sys
from dataclasses import dataclass, field, replace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CARDS = os.path.join(ROOT, "results", "kda-ceiling-cards-20261013", "cards.json")
CU, CLK = 80, 1500.0                       # CUs; cycles per microsecond
HBM_RD, HBM_WR = 1339.2e3, 1100.5e3        # bytes per microsecond (kra calibration, bw7 exclusive recheck)
LDS_CU, VGPR_SIMD, WAVES_CU = 65536, 768, 40
CPI_VALU, CPI_TRANS = 8.0, 16.0            # cycles an added dependent VALU / transcendental adds to a CTA
BETA_MEM = 0.5                             # share of a removed stream's latency that leaves the critical path
E_PHYS = 1.10                              # attainable throughput efficiency of a latency-bound kernel's floors
CPI_MMAC = 16.0                            # cycles an added dependent 16x16x16 MMAC adds to a CTA
KAPPA_TA = 0.8                             # share of a CTA's added TA cycles on its latency (calibrated, P-20)
# TA cost table (results/kda-bound-20261012, CU cycles per wave-level VMEM instruction, data in HBM)
TA_RD = {"c1k": 21, "c512": 22, "r4x256": 43, "r16x64": 169, "r16x32": 85, "gather64": 229, "c256": 10,
         "gather4": 43}   # gather4: s4b aqB / akB = 4 B/lane in 4 rows x 64 B per instruction (read from the code;
                          # the first step-2 run assumed the 64 rows x 4 B pattern, 229 cycles -> P-55 / L1 / L2 overrated)
TA_WR = {"c1k": 56, "c512": 28, "r16x64": 67, "r16x32": 67}
FLOORS = ("f_mem", "f_ta", "f_val", "f_lds", "f_cmp", "f_chain")


@dataclass
class Kernel:
    name: str
    ctas: int
    waves: int                 # waves per CTA
    lds: int                   # bytes per CTA
    vgpr: int
    lam: float                 # us per CTA (latency term)
    e: float                   # efficiency of the throughput term
    f: dict = field(default_factory=dict)       # stream floors, us
    rd: float = 0.0            # HBM bytes read / written
    wr: float = 0.0
    nw: float = 0.0            # instructions per wave (VALU + VMEM + LDS, PMC)
    vmem_w: float = 0.0        # VMEM instructions per wave
    res_cap: int = 0           # residency cap imposed by the launcher (P-24 throttles); 0 = none
    note: str = ""
    regime: str = "thr"
    t_meas: float = 0.0

    def resident(self):
        r = min(LDS_CU // max(self.lds, 1), WAVES_CU // self.waves,
                4 * max(1, VGPR_SIMD // max(self.vgpr, 1)) // self.waves)
        if self.res_cap:
            r = min(r, self.res_cap)
        return max(r, 0)

    def rounds(self):
        r = self.resident()
        return math.inf if r == 0 else math.ceil(self.ctas / (CU * r))

    def F(self):
        return max(self.f.values())

    def imb(self):
        # single partial wave of CTAs (<= 80): the cards' floors are per active CU already
        return 1.0 if self.ctas <= CU else math.ceil(self.ctas / CU) / (self.ctas / CU)

    def t(self):
        R = self.rounds()
        return max(R * self.lam, self.F() * self.imb() * self.e)

    def feasible(self):
        return self.resident() > 0 and self.vgpr <= 256


def load(shape):
    alld = json.load(open(CARDS))
    d = alld[shape]["cards"]
    T = shape.split("_")[0]
    iso = alld.get(f"{T}_12", {}).get("cards", {})
    ks = {}
    for n, c in d.items():
        waves = c["block"] // 64
        k = Kernel(name=n, ctas=c["ctas"], waves=waves, lds=c["lds_c"], vgpr=c["vgpr"], lam=0.0, e=1.0,
                   f={x: c[x] for x in FLOORS}, rd=c["hbm_rd"], wr=c["hbm_wr"],
                   nw=(c["valu_insts"] + c["vmem"] + c["lds_insts"]) / (c["ctas"] * waves),
                   vmem_w=c["vmem"] / (c["ctas"] * waves))
        # the launcher's residency (P-24 throttle / cards) overrides the generic rule when lower
        if c["resident"] < k.resident():
            k.res_cap = c["resident"]
        k.lam = c["t_us"] / k.rounds()
        if k.rounds() == 1 and k.ctas > CU and n in iso and iso[n]["ctas"] <= CU:
            k.lam = min(k.lam, iso[n]["t_us"])          # isolated per-CTA latency (same kernel at H = 12)
        # regime: a latency-bound kernel (gap > 1.2 and the latency term explains t) keeps a PHYSICAL throughput
        # efficiency (E_PHYS) so that latency savings show; a throughput-bound one calibrates e to t
        k.e = c["t_us"] / (k.F() * k.imb())
        k.regime = "thr"
        if c["gap"] > 1.2 and k.rounds() * k.lam >= 0.98 * c["t_us"] and k.e > E_PHYS:
            k.e, k.regime = E_PHYS, "lat"
        k.t_meas = c["t_us"]
        ks[n] = k
    return ks


def cpi(k):
    """measured cycles per wave-instruction of k (latency term)."""
    return k.lam * CLK / max(k.nw, 1.0)


def add_work(k, valu=0.0, trans=0.0, vmem=0.0, rd=0.0, wr=0.0, valu_slots_us=0.0, ta_us=0.0, ta_cyc=0.0,
             mmac=0.0, note=""):
    """per-WAVE added instructions (valu / transcendental / vmem / mmac) and TA cycles (TA cost table x pattern),
    whole-kernel added bytes / floors.  A wave's TA cycles reach its CTA's latency with KAPPA_TA (P-20)."""
    k2 = replace(k, f=dict(k.f), note=(k.note + " " + note).strip())
    k2.lam = k.lam + (valu * CPI_VALU + trans * CPI_TRANS + vmem * cpi(k) + mmac * CPI_MMAC
                      + ta_cyc * k.waves * KAPPA_TA) / CLK
    ta_us = ta_us + ta_cyc * k.waves * k.ctas / (CU * CLK * 0.92)
    k2.rd, k2.wr = k.rd + rd, k.wr + wr
    k2.f["f_mem"] = k2.rd / HBM_RD + k2.wr / HBM_WR
    k2.f["f_val"] = k.f["f_val"] + valu_slots_us
    k2.f["f_ta"] = k.f["f_ta"] + ta_us
    k2.f["f_cmp"] = k.f["f_cmp"] + mmac * k.waves * k.ctas * 8192 / (476e6 * 0.92)   # bf16 MMAC FLOP/us
    k2.nw = k.nw + valu + trans + vmem + mmac
    return k2


def remove_bytes(k, rd=0.0, wr=0.0, note=""):
    """dropped HBM traffic: floors shrink by the bytes (TA by the kernel's own TA-per-byte), lam by BETA_MEM x
    the traffic's share."""
    k2 = replace(k, f=dict(k.f), note=(k.note + " " + note).strip())
    tot = k.rd + k.wr
    share = (rd + wr) / tot if tot else 0.0
    k2.rd, k2.wr = k.rd - rd, k.wr - wr
    k2.f["f_mem"] = k2.rd / HBM_RD + k2.wr / HBM_WR
    k2.f["f_ta"] = k.f["f_ta"] * (1 - share)
    k2.lam = k.lam * (1 - BETA_MEM * share)
    return k2


def merge(host, guest, lds_extra=0, vgpr_extra=0, vgpr=None, name=None):
    """guest's work runs inside host's CTAs (same grid).  Traffic between them must be removed separately
    (remove_bytes) before / after."""
    assert host.ctas == guest.ctas, "merge needs the same grid (C2)"
    f = {x: host.f[x] + guest.f[x] for x in FLOORS}
    f["f_chain"] = max(host.f["f_chain"], guest.f["f_chain"])
    k = replace(host, name=name or f"{host.name}+{guest.name}", f=f, lds=host.lds + lds_extra,
                vgpr=vgpr if vgpr is not None else max(host.vgpr, guest.vgpr) + vgpr_extra,
                lam=host.lam + guest.lam, rd=host.rd + guest.rd, wr=host.wr + guest.wr,
                nw=host.nw + guest.nw, e=host.e)
    # the guest's latency was measured at ITS residency; if the merged kernel holds fewer CTAs per CU the
    # rounds() term carries the occupancy loss
    return k


def geometry(k, cta_factor, work_factor, lds=None, vgpr=None, res_cap=None, floor_factor=None):
    """wider / narrower CTAs: ctas x cta_factor, per-CTA work x work_factor (latency), whole-kernel floors x
    floor_factor (default cta_factor x work_factor)."""
    ff = cta_factor * work_factor if floor_factor is None else floor_factor
    k2 = replace(k, ctas=int(round(k.ctas * cta_factor)), lam=k.lam * work_factor,
                 f={x: v * ff for x, v in k.f.items()}, lds=lds or k.lds, vgpr=vgpr or k.vgpr,
                 res_cap=k.res_cap if res_cap is None else res_cap)
    return k2


# ----------------------------------------------------------------------------------------------- back-tests
def bt_b4b():
    """B4b-lite (P-47/47b/47c, results/kda-b4-design-20261012): bwd_tail inside s4b, 4 waves / 2 CTAs per CU kept,
    dq/dk still written fp32 and read back at the kernel end (traffic kept), dg gate-backward in the kt loop
    (the fp32 dg write + read removed), tail LDS reuses s4b's.  Measured 16K/TP8: fused 2160-2267 us vs
    s4b + tail + reduce 2017 us -> 1.07-1.12x slower."""
    ks = load("16384_12")
    s4b, tail, red = ks["wy_s4b#1"], ks["bwd_tail#1"], ks["bwd_tail_reduce#1"]
    dg = 16384 * 12 * 128 * 4
    host = remove_bytes(s4b, wr=dg)
    guest = remove_bytes(tail, rd=dg)
    m = merge(host, guest, lds_extra=0, vgpr=252)
    base = s4b.t() + tail.t() + red.t()
    pred = base / (m.t() + red.t())
    return dict(name="B4b-lite: bwd_tail fused into s4b (16K/12)", pred=pred, meas=(1 / 1.12, 1 / 1.07),
                detail=f"fused {m.t():.0f} us (R={m.rounds()} x lam {m.lam:.1f}; F*e {m.F() * m.e:.0f}) vs "
                       f"{base:.0f}")


def bt_p13(shape, meas):
    """P-13b (validation/results.md): l2n_apply removed, consumers recompute bf16(x*rstd): ~24 instructions per
    quad (bf2f, mul, software RNE, pack); wu 3 sites, s4b every kt step, tail every row.  Per wave:
    wu 3 sites x 8 quads, s4b 8 kt x 2 tensors, tail 16 rows x 2 tensors x 2 elems / 4."""
    ks = load(shape)
    T, H = map(int, shape.split("_"))
    l2n, wu, s4b, tail = ks["l2n_apply#1"], ks["wu#1"], ks["wy_s4b#1"], ks["bwd_tail#1"]
    rstd = T * H * 4
    wu2 = add_work(wu, valu=3 * 8 * 24, rd=rstd)
    s4b2 = add_work(s4b, valu=8 * 2 * 24, rd=rstd)
    tail2 = add_work(tail, valu=16 * 24, rd=rstd)
    tot = sum(k.t() for k in ks.values())
    new = tot - l2n.t() - wu.t() - s4b.t() - tail.t() + wu2.t() + s4b2.t() + tail2.t()
    return dict(name=f"P-13b: l2n_apply -> consumer recompute ({shape})", pred=tot / new, meas=meas,
                detail=f"l2n -{l2n.t():.0f}, wu +{wu2.t() - wu.t():.0f}, s4b +{s4b2.t() - s4b.t():.0f}, "
                       f"tail +{tail2.t() - tail.t():.0f} us (op speedup = pred)")


def bt_p51(shape, meas):
    """P-51 (results/kda-gfx936-20261013-p51): dhu 4-wave with 64 columns per CTA: CTAs halve, per-CTA work
    ~1.8x (MMACs, dv2 / bN / dh emits double, shared K-side reads once), LDS 41 KB -> 1 CTA per CU (as the
    production P-24 throttle).  Kernel ratio production / wide."""
    ks = load(shape)
    d = ks["dhu_v8b4#1"]
    w = geometry(d, 0.5, 1.8, lds=41 * 1024, res_cap=1, floor_factor=0.9)
    return dict(name=f"P-51: dhu 64 columns per CTA ({shape})", pred=d.t() / w.t(), meas=meas,
                detail=f"R {d.rounds()} -> {w.rounds()}, lam {d.lam:.0f} -> {w.lam:.0f} us")


def bt_p49(shape, meas, shared=(0.5, 0.4)):
    """P-49 (results/kda-gfx936-20261013-r13b): forward / recompute recurrence with 64 columns per CTA (2 CTAs per
    head instead of 4): per-CTA work ~1.8x, the K-side operands (share s of a narrow CTA's demand) are loaded
    by 2 CTAs instead of 4 -> whole-kernel demand x (1 - s/2).  Kernel ratios production / wide (#1, #2)."""
    ks = load(shape)
    out, det = [], []
    for name, s_ in zip(("fwd_h_v8o4#1", "fwd_h_v8o4#2"), shared):
        if name in ks:                                  # production = narrow: predict the wide kernel
            k = ks[name]
            w = geometry(k, 0.5, 1.8, floor_factor=1 - s_ / 2, res_cap=2)
        else:                                           # production = wide (H >= 40, NT >= 512): predict narrow
            w = ks[name.replace("v8o4", "v8o4w")]
            k = geometry(w, 2.0, 1 / 1.8, floor_factor=1 / (1 - s_ / 2), res_cap=4)
        out.append(k.t() / w.t())
        det.append(f"{name[-2:]}: R {k.rounds()}->{w.rounds()} imb {k.imb():.2f}->{w.imb():.2f} [{k.regime}]")
    return dict(name=f"P-49: recurrences 64 columns per CTA ({shape}) #1/#2", pred=out, meas=meas,
                detail="; ".join(det))


def bt_p53(shape, meas):
    """P-53: forward recurrence keeps the GEMM1 h fragments for the o pass: -8 ds_read2_b64 per wave per step
    (charged like a dependent VALU-class issue, CPI_VALU)."""
    ks = load(shape)
    k = ks["fwd_h_v8o4#1"]
    NT = int(shape.split("_")[0]) // 64
    k2 = add_work(k, valu=-8 * NT)
    k2.f["f_lds"] = k.f["f_lds"] * (1 - 8 / 26)
    return dict(name=f"P-53: fwd h-fragment reuse ({shape}) fwd#1", pred=k.t() / k2.t(), meas=meas,
                detail=f"lam {k.lam:.0f} -> {k2.lam:.0f} us")


def bt_p20(shape, meas):
    """P-20 (validation/results.md): dav stops writing the aN_do copy (T*H*128 bf16, scattered 8 B stores);
    s4b reads do row-major: same 8 ld8 per lane, pattern c512 -> r16x32 (+63 TA cycles each).
    Kernel ratios dav, s4b (old / new); built backwards from today's cards (= the P-20 state)."""
    ks = load(shape)
    T, H = map(int, shape.split("_"))
    dav, s4b = ks["dav#1"], ks["wy_s4b#1"]
    cp = T * H * 128 * 2
    dav_old = add_work(dav, wr=cp, vmem=4, ta_cyc=4 * TA_WR["r16x32"])
    s4b_old = add_work(s4b, ta_cyc=-8 * (TA_RD["r16x32"] - TA_RD["c512"]))
    return dict(name=f"P-20: drop aN_do, s4b reads do row-major ({shape}) dav/s4b", fmt="ratio",
                pred=[dav_old.t() / dav.t(), s4b_old.t() / s4b.t()], meas=meas,
                detail=f"dav {dav_old.t():.0f}->{dav.t():.0f}, s4b {s4b_old.t():.0f}->{s4b.t():.0f} us")


def bt_p17(shape, meas):
    """P-17: wu stops writing vB (T*H*128 bf16, scattered 8 B stores); s4a builds the operand on chip:
    -32 native loads (c512) +32 LDS reads, +4 row-major v loads (r4x256) +16 LDS b16 writes + 1 barrier."""
    ks = load(shape)
    T, H = map(int, shape.split("_"))
    wu, s4a = ks["wu#1"], ks["wy_s4a#1"]
    cp = T * H * 128 * 2
    wu_old = add_work(wu, wr=cp, vmem=4, ta_cyc=4 * TA_WR["r16x32"])
    s4a_old = add_work(s4a, valu=-32 - 16 - 30, ta_cyc=32 * TA_RD["c512"] - 4 * TA_RD["r4x256"])
    return dict(name=f"P-17: drop vB, s4a builds v on chip ({shape}) wu/s4a", fmt="ratio",
                pred=[wu_old.t() / wu.t(), s4a_old.t() / s4a.t()], meas=meas,
                detail=f"wu {wu_old.t():.0f}->{wu.t():.0f}, s4a {s4a_old.t():.0f}->{s4a.t():.0f} us")


def bt_f6():
    """F6 (P-39): s4a and s4b in one kernel with both LDS working sets (30 KB + 32 KB) and s4b's 236 VGPR."""
    ks = load("16384_12")
    a, b = ks["wy_s4a#1"], ks["wy_s4b#1"]
    m = merge(b, a, lds_extra=a.lds, vgpr=256)
    return dict(name="F6: s4a + s4b in one kernel (16K/12)", pred=(a.t() + b.t()) / m.t() if m.feasible() else 0,
                meas=("infeasible or slower",),
                detail=f"LDS {m.lds / 1024:.1f} KB -> resident {m.resident()}, R {m.rounds()} "
                       f"(was {b.rounds()}): fused {m.t():.0f} vs {a.t() + b.t():.0f} us")


CPI_STORE_CHAIN = 205.0   # cycles per added VMEM store inside a latency-bound recurrence step (calibrated, P-50)


def bt_p50_store():
    """Stores on a latency-bound recurrence chain cost far more than their issue / TA share (the step waits for
    them).  Calibrated on P-50 keep-h (forward MODE 5: +4 bN stores per wave per step -> +140 us at 16K/12),
    checked on the full keep-state forward (h + Vn + aN: 4 + 2 + 2 stores per wave per step -> +262 us)."""
    ks = load("16384_12")
    k = ks["fwd_h_v8o4#1"]
    NT = 256
    out = []
    for nst in (4, 8):
        k2 = replace(k, lam=k.lam + nst * NT * CPI_STORE_CHAIN / CLK)
        out.append(k2.t() - k.t())
    return dict(name="P-50 / keep-state: forward stores on the chain (16K/12) us", pred=out,
                meas=((140.0, 140.0), (262.0, 262.0)), detail="keep-h calibrates CPI_STORE_CHAIN; full keep-state checks it",
                fmt="us")


def regimes():
    """kernel x shape: regime (lat / thr), rounds, busiest-CU imbalance -- straight from the cards."""
    shapes = [f"{T}_{H}" for T in (8192, 16384, 32768, 65536) for H in (12, 24, 48, 96)]
    allk = []
    data = {}
    for sh in shapes:
        ks = load(sh)
        data[sh] = ks
        for n in ks:
            if n not in allk and ks[n].t_meas >= 20:
                allk.append(n)
    print(f"{'kernel':18s} " + " ".join(f"{sh.replace('_', '/'):>9s}" for sh in shapes))
    for n in allk:
        cells = []
        for sh in shapes:
            k = data[sh].get(n)
            cells.append(f"{k.regime}R{k.rounds()}" if k else "-")
        print(f"{n:18s} " + " ".join(f"{c:>9s}" for c in cells))
    print("lat = latency/issue-bound (fusion adds latency, occupancy matters); thr = throughput-bound (floors bind)")


# ------------------------------------------------------------------------------------------- candidates
# s4b prologue per wave (averaged over the 4 waves, A = 0..3; g2_wy.cuh s4b_tail): aqA (A+1) f4 rows (r16x64),
# aqB (4-A)x4 scalar gathers of dAqk (gather4), akB (4-A)x4 scalar gathers of s4a's dA, dak 4 f4 rows.
S4B_AQA, S4B_AQB, S4B_AKB = 2.5, 10.0, 10.0
ADDR_PER_GATHER = 3          # 64-bit address VALU per scalar gather (j * RA + row)


def cand_f3a(ks, T, H):
    """F3a: s4b recomputes dAqk = scale * do . Vn^T (causal) for its own 16 rows x 64 columns and the transposed
    form, from the do fragments it already loads (ado) + Vn of the other 3 row blocks (24 ld8, c512);
    48 MMAC per wave.  dav loses pass 1 (Vn staging loads, 32 MMAC, dA LDS staging, fp32 dAqk stores) and the
    Vn read; nobody writes / reads dAqk.  VGPR: s4b at 236 + ~48 -> spill risk (B4b lesson)."""
    dav, s4b = ks["dav#1"], ks["wy_s4b#1"]
    vn, daq = T * H * 128 * 2, T * H * 64 * 4
    dav2 = remove_bytes(dav, rd=vn, wr=daq)
    dav2 = add_work(dav2, mmac=-32, vmem=-(8 + 8 + 4), valu=-(32 + 16 + 8), ta_cyc=-(8 * TA_RD["c512"] + 4 * TA_WR["c1k"]))
    s4b2 = remove_bytes(s4b, rd=daq)
    s4b2 = add_work(s4b2, mmac=48, vmem=24 - S4B_AQA - S4B_AQB, valu=16 - ADDR_PER_GATHER * S4B_AQB,
                    ta_cyc=24 * TA_RD["c512"] - S4B_AQA * TA_RD["r16x64"] - S4B_AQB * TA_RD["gather4"],
                    rd=0.0)
    s4b2.vgpr = 256
    return {"dav#1": dav2, "wy_s4b#1": s4b2}, "s4b VGPR 236+48 > 256: spill risk", "A (same MMAC order as dav)"


def cand_tcopy(ks, T, H, which=("dav", "s4a")):
    """L1/L2: the producers also write a TRANSPOSED fp32 copy (dav: dAqk^T from its LDS staging; s4a: dA^T, LDS
    staging reused at the kernel end), so s4b reads aqB / akB as f4 rows (r16x64) instead of 4 B gathers."""
    out = {}
    cp = T * H * 64 * 4
    s4b = ks["wy_s4b#1"]
    n_g, n_r = 0.0, 0.0
    if "dav" in which:
        out["dav#1"] = add_work(ks["dav#1"], wr=cp, vmem=4, valu=16, ta_cyc=4 * TA_WR["c1k"])
        n_g += S4B_AQB; n_r += S4B_AQB / 4
    if "s4a" in which:
        out["wy_s4a#1"] = add_work(ks["wy_s4a#1"], wr=cp, vmem=4, valu=16 + 1, ta_cyc=4 * TA_WR["c1k"])
        n_g += S4B_AKB; n_r += S4B_AKB / 4
    out["wy_s4b#1"] = add_work(s4b, rd=0.0, vmem=n_r - n_g, valu=-ADDR_PER_GATHER * n_g,
                               ta_cyc=n_r * TA_RD["r16x64"] - n_g * TA_RD["gather4"])
    return out, "LDS reuse in s4a assumed (else 46 KB -> 1 CTA/CU)", "A (layout only)"


def cand_f2_wb(ks, T, H):
    """F2 (representative): wu stops writing Wb (B form of W); dhu reads Wn and transposes through LDS each step
    (+16 LDS writes, +16 LDS reads, +1 barrier per wave per step)."""
    wu = ks["wu#1"]
    d = ks.get("dhu_v8b#1") or ks.get("dhu_v8b4#1")
    dn = "dhu_v8b#1" if "dhu_v8b#1" in ks else "dhu_v8b4#1"
    NT = T // 64
    cp = T * H * 128 * 2
    wu2 = add_work(wu, wr=-cp, vmem=-4, ta_cyc=-4 * TA_WR["r16x32"])
    d2 = add_work(d, valu=NT * (32 + 8))      # 32 LDS ops + barrier (~8 issue-equivalents) per step
    return {"wu#1": wu2, dn: d2}, "", "A"


def cand_gate_wu(ks, T, H):
    """wu computes g itself from g_raw (bf16, 2 B instead of fp32 4 B) with a per-column sequential chunk scan
    (64 dependent adds + sigmoid per element: exp + rcp); the gate kernel stays for s4 / to_gn."""
    wu = ks["wu#1"]
    wu2 = add_work(wu, rd=-T * H * 128 * 2, trans=2 * 32, valu=64 + 6 * 32)
    return {"wu#1": wu2}, "", "A if the scan order equals the gate kernel's"


def cand_dav_dhu(ks, T, H):
    """dav's dv = A^T do (V-separable) inside the dh recurrence: per step per wave +8 MMAC, A^T fragments as
    4 B gathers (32 per wave per step) + the do slab; dVn / dOT writes + reads go; dav keeps pass 1."""
    dav = ks["dav#1"]
    d = ks.get("dhu_v8b#1") or ks.get("dhu_v8b4#1")
    dn = "dhu_v8b#1" if "dhu_v8b#1" in ks else "dhu_v8b4#1"
    NT = T // 64
    two = 2 * T * H * 128 * 2
    dav2 = remove_bytes(dav, wr=two)
    dav2 = add_work(dav2, mmac=-32, vmem=-12, valu=-40)
    d2 = remove_bytes(d, rd=two)
    d2 = add_work(d2, mmac=8 * NT, ta_cyc=32 * TA_RD["gather4"] * NT, vmem=32 * NT)
    return {"dav#1": dav2, dn: d2}, "", "A"


CANDS = [("F3a  s4b recomputes dAqk (dav pass 1 goes)", cand_f3a),
         ("L1   dav writes dAqk^T; s4b aqB as rows", lambda ks, T, H: cand_tcopy(ks, T, H, ("dav",))),
         ("L2   s4a writes dA^T; s4b akB as rows", lambda ks, T, H: cand_tcopy(ks, T, H, ("s4a",))),
         ("L1+L2 both transposed copies", lambda ks, T, H: cand_tcopy(ks, T, H, ("dav", "s4a"))),
         ("F2   drop Wb, dhu transposes Wn in LDS", cand_f2_wb),
         ("G    wu computes g from g_raw", cand_gate_wu),
         ("D    dav's dv inside the dh recurrence", cand_dav_dhu)]


def candidates():
    shapes = [f"{T}_{H}" for T in (8192, 16384, 32768, 65536) for H in (12, 24, 48, 96)]
    print(f"{'candidate':42s} " + " ".join(f"{sh.replace('_', '/'):>8s}" for sh in shapes) + "   TP8   TP2   mean")
    rows = []
    for name, fn in CANDS:
        cells, by_h = [], {12: [], 24: [], 48: [], 96: []}
        note = cls = ""
        for sh in shapes:
            T, H = map(int, sh.split("_"))
            ks = load(sh)
            base = sum(k.t() for k in ks.values())
            new_k, note, cls = fn(ks, T, H)
            feas = all(k.feasible() for k in new_k.values())
            new = base - sum(ks[n].t() for n in new_k) + sum(k.t() for k in new_k.values())
            g = 100.0 * (base / new - 1.0)
            cells.append(f"{g:+7.2f}%" if feas else "  infeas")
            by_h[H].append(g)
        m8, m2 = sum(by_h[12]) / 4, sum(by_h[48]) / 4
        allg = [x for v in by_h.values() for x in v]
        rows.append((name, cells, m8, m2, sum(allg) / len(allg), note, cls))
        print(f"{name:42s} " + " ".join(f"{c:>8s}" for c in cells) + f" {m8:+5.2f} {m2:+5.2f} {sum(allg) / len(allg):+5.2f}")
    print()
    for name, _, _, _, _, note, cls in rows:
        print(f"  {name:42s} class: {cls:30s} {note}")
    print("(operator-level speedup %, positive = faster; model is conservative for copy removals, see backtest)")


def backtest():
    rows = [bt_b4b(),
            bt_p13("8192_12", (1.002, 1.002)), bt_p13("8192_48", (1.005, 1.005)),
            bt_p51("16384_48", (0.84, 0.86)), bt_p51("32768_48", (0.87, 0.89)), bt_p51("65536_48", (0.87, 0.87)),
            bt_f6(),
            bt_p49("16384_48", ((1.01, 1.01), (0.91, 0.91))), bt_p49("32768_48", ((1.12, 1.12), (1.02, 1.02))),
            bt_p49("65536_48", ((1.40, 1.40), (1.12, 1.12))), bt_p49("8192_96", ((0.90, 0.90), (0.68, 0.68))),
            bt_p53("8192_12", (1.018, 1.018)), bt_p53("16384_12", (1.012, 1.012)), bt_p53("32768_12", (1.010, 1.010)),
            bt_p53("8192_48", (1.020, 1.020)), bt_p53("16384_48", (1.018, 1.056)), bt_p53("8192_96", (1.025, 1.025)),
            bt_p50_store(),
            bt_p20("8192_12", ((1.148, 1.148), (0.983, 0.983))), bt_p20("16384_48", ((1.247, 1.247), (0.996, 0.996))),
            bt_p20("8192_96", ((1.396, 1.396), (1.008, 1.008))),
            bt_p17("8192_12", ((1.052, 1.052), (1.045, 1.045))), bt_p17("16384_48", ((1.081, 1.081), (1.055, 1.055))),
            bt_p17("8192_96", ((1.208, 1.208), (1.033, 1.033)))]
    print(f"{'experiment':56s} {'pred':>13s} {'measured':>21s}  detail")
    for r in rows:
        m, p = r["meas"], r["pred"]
        if isinstance(p, list) and r.get("fmt") == "us":
            ps = " / ".join(f"{x:.0f}" for x in p)
            ms = " / ".join(f"{x[0]:.0f}" for x in m)
        elif isinstance(p, list):
            ps = " / ".join(f"{x:.3f}" for x in p)
            ms = " / ".join(f"{x[0]:.2f}" if x[0] == x[1] else f"{x[0]:.2f}-{x[1]:.2f}" for x in m)
        else:
            ps = f"{p:.3f}"
            ms = (f"{m[0]:.3f}-{m[1]:.3f}" if m[0] != m[1] else f"{m[0]:.3f}") if isinstance(m[0], float) else m[0]
        print(f"{r['name']:56s} {ps:>13s} {ms:>21s}  {r['detail']}")
    print("(ratios: >1 = the change is faster; B4b / P-51 / F6 are expected < 1)")


if __name__ == "__main__":
    {"backtest": backtest, "regimes": regimes, "candidates": candidates}[sys.argv[1] if len(sys.argv) > 1 else "backtest"]()
