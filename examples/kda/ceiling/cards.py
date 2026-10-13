"""Ceiling cards: per-(kernel, shape) resource-limit diagnosis.

Floor = max(mem, compute, issue, chain); the binding term names the resource.
Work gaps = actual / required (bytes, vmem instructions, FLOPs are absolute).
Output: cards.json + cards.md  (draft for 16K x 4 shapes first).

    python3 cards.py            # all shapes present in raw/
    python3 cards.py 16384_48   # one shape
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from parse_raw import parse_shape                                  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "results" / "kda-ceiling-cards-20261013"
RAW = RES / "raw"
MACH = json.load(open(ROOT / "machines" / "gfx936-bw1000.json"))
ISA = json.load(open(RAW / "isa_mix.json")) if (RAW / "isa_mix.json").exists() else {}
CLK = 1.5e9
P = MACH["peaks"]
HBW_R, HBW_W = P["hbm_read"]["value"] * 1e9, P["hbm_write"]["value"] * 1e9
L2_BW, LDS_BW = P["l2_read"]["value"] * 1e9, P["lds_read"]["value"] * 1e9
MMAC_PK = P["mmac_bf16"]["value"] * 1e12 * 0.92          # attainable (GEMM1-style) 0.92
VALU_PK = 80 * 1 * CLK                                  # 1 wave64 valu inst / CU / cycle
TA_PK = 80 * CLK                                        # 1 issue slot / CU / cycle (upper)

# measured bound_microbench.hip k_step (us/step): {with-o, state-only} at 48 and 192 CTAs.
# Idle-GPU numbers from results/kda-bound-20261012; a rerun under shared-GPU pressure
# (HCU5, other tenants) read 1.26/1.26 @48 and 3.29/1.74 @192 -- ~25% high, treat as upper.
LSTEP = {"1cu": (1.241e-6, 1.015e-6), "2cu": (2.557e-6, 1.729e-6)}
MMAC_FLOPS = {"v_mmac_f32_16x16x16_bf16": 8192, "v_mmac_16x16x8_f32": 4096,
              "v_mmac_f32_16x16x16_f16": 8192, "v_mmac_f32_16x16x8_tf32": 4096,
              "v_mmac_i32_16x16x32_i8": 16384}

# production kernel -> mangled template fragment (GE=false production paths)
MANGLED = {
    "fwd_prep_a": "fwd_prep_a_kernelILi1ELb0", "fwd_prep_b": "fwd_prep_b_kernelILb0",
    "fwd_h_v8o4#1": "fwd_h_v8o4_kernelILb0ELi1ELb1", "fwd_h_v8o4#2": "fwd_h_v8o4_kernelILb0ELi4ELb0",
    "wu": "g2_wu_kernelILb0", "dav": "g2_dav_kernelILb0",
    "dhu_v8b4": "dhu_v8b4_kernelILi0ELb0", "dhu_v8b2": "dhu_v8b_kernelILi2ELb0",
    "dhu_v8b": "dhu_v8b_kernelILi0ELb0",
    "wy_s4a": "p3_wy_s4a_kernelILb0ELb0", "wy_s4b": "p3_wy_s4b_kernelILb0",
    "bwd_tail": "g2_bwd_tail_kernelILb0", "bwd_tail_reduce": "g2_bwd_tail_reduce",
    "gate": "g2_gate_kernelILi1ELb0", "l2n_apply": "g2_l2n_apply", "exp2": "g2_exp2_kernel",
    "elementwise": None,
}
# launcher -> member kernels (required-byte split by measured-byte share, approx)
LAUNCHER_MEMBERS = {
    "fwd_prep": ["fwd_prep_a#1", "fwd_prep_b#1"], "fwd_h_o": ["fwd_h_v8o4#1", "fwd_h_v8o4w#1"],
    "l2n_apply": ["l2n_apply#1"], "head_gate": ["gate#1", "elementwise_manual_unroll#1",
                                              "elementwise_manual_unroll#2", "exp2#1"],
    "wu": ["wu#1"], "fwd_h_bn": ["fwd_h_v8o4#2", "fwd_h_v8o4w#2"], "dav": ["dav#1"],
    "dhu_bn": ["dhu_v8b4#1", "dhu_v8b#1", "dhu_v8b2#1"], "wy_s4_split": ["wy_s4a#1", "wy_s4b#1"],
    "bwd_tail": ["bwd_tail#1", "bwd_tail_reduce#1"],
}
RECURRENCE = {"fwd_h_v8o4#1": "with_o", "fwd_h_v8o4#2": "state", "dhu_v8b4#1": "state"}


def isa_of(short: str) -> dict:
    frag = MANGLED.get(re.sub(r"#\d+$", "", short) if "#" in short else short)
    frag = MANGLED.get(short, frag)
    if not frag:
        return {}
    for n, v in ISA.items():
        if frag in n:
            return v
    return {}


def mmac_flops(key: str, mmop: float) -> tuple[float, str]:
    """FLOPs from executed MMAC instruction count, typed via ISA."""
    if not mmop:
        return 0.0, "none"
    rec = isa_of(key)
    mm = rec.get("mmac", {})
    tot = sum(mm.values())
    if not tot:
        return mmop * 8192.0, "no-isa, bf16 assumed"
    fl = sum(MMAC_FLOPS.get(k, 8192) * v for k, v in mm.items()) / tot
    pure = "pure bf16" if set(mm) == {"v_mmac_f32_16x16x16_bf16"} else \
           "pure f32x8" if set(mm) == {"v_mmac_16x16x8_f32"} else f"mix {mm}"
    return mmop * fl, pure


def card(shape: str) -> dict:
    T, H = map(int, shape.split("_"))
    NT = T // 64
    r = parse_shape(RAW / shape)
    launchers = r["launchers"]
    cards = {}
    # measured-byte share per launcher for required-byte split
    memb_meas = {ln: sum((r["kernels"].get(m, {}).get("hbm_rd", 0) +
                          r["kernels"].get(m, {}).get("hbm_wr", 0)) for m in mem)
                 for ln, mem in LAUNCHER_MEMBERS.items()}
    for key, v in r["kernels"].items():
        t = v.get("t_us")
        if not t:
            continue
        fl, mkind = mmac_flops(key, v["mmop"])
        hbm = v["hbm_rd"] + v["hbm_wr"]
        l2 = v["tcc_req"] * 64
        lds = v["lds_idx"] * 128
        t_s = t * 1e-6
        # ---- floors (per-CU resources normalized by concurrently active CUs)
        act = min(80, v["ctas"])
        f_mem = v["hbm_rd"] / HBW_R + v["hbm_wr"] / HBW_W
        f_cmp = fl / (MMAC_PK * act / 80) if fl else 0.0
        f_ta = v["ta_busy"] / (act * CLK)                       # measured TA-busy time
        f_val = v["valu_slots"] / (act * CLK)                   # 1 valu slot / CU / cycle
        f_lds = v["lds_idx"] / (act * CLK)                      # IDX busy incl. replays
        f_issue = max(f_ta, f_val, f_lds)
        f_chain = 0.0
        if key in RECURRENCE:
            w_o = RECURRENCE[key] == "with_o"
            l = LSTEP["1cu"] if v["ctas"] <= 84 else LSTEP["2cu"]
            f_chain = NT * (l[0] if w_o else l[1])
        floor = max(f_mem, f_cmp, f_issue, f_chain)
        binding = max((f_mem, "mem"), (f_cmp, "compute"), (f_issue, "issue"), (f_chain, "chain"))[1]
        # ---- required bytes (launcher interface, split by measured share for multi-kernel)
        req = None
        for ln, mem in LAUNCHER_MEMBERS.items():
            if key in mem:
                lb = launchers.get(ln)
                if lb:
                    tot = lb["in_bytes"] + lb["out_bytes"]
                    if len(mem) > 1:
                        mb = (v["hbm_rd"] + v["hbm_wr"])
                        share = mb / memb_meas[ln] if memb_meas[ln] else 1.0 / len(mem)
                        req = tot * share
                    else:
                        req = tot
        # ---- occupancy (LDS cap, wave-slot cap, measured 3-waves/SIMD rule R38)
        lds_c, block = v["lds_c"], v["block"]
        waves = block // 64
        res = min(64 * 1024 // max(1, lds_c), 12 // max(1, waves), 40 // max(1, waves))
        resident = max(1, res)
        rounds = -(-v["ctas"] // (80 * resident))
        act_n = v["grbm"] * act if v["grbm"] else 1
        gap = t_s / floor if floor else None
        cards[key] = dict(
            t_us=t, spread=v.get("spread"), procs=v.get("procs"),
            ctas=v["ctas"], block=block, lds_c=lds_c, vgpr=v["vgpr"], sgpr=v["sgpr"],
            flops=fl, mmac_kind=mkind, hbm_rd=v["hbm_rd"], hbm_wr=v["hbm_wr"],
            l2=l2, lds=lds, vmem=v["vmem_rd"] + v["vmem_wr"], tcc_req=v["tcc_req"],
            valu_i=v["valu_insts"], valu_s=v["valu_slots"], mmop=v["mmop"],
            lds_conf=v["lds_conf"], lds_wait=v["lds_wait"], lds_insts=v["lds_insts"],
            lds_idx=v["lds_idx"], valu_insts=v["valu_insts"], valu_slots=v["valu_slots"],
            mmop_active=v["mmop_active"],
            ta_busy=v["ta_busy"], grbm=v["grbm"],
            hbm_gbs=hbm / t_s / 1e9, l2_gbs=l2 / t_s / 1e9, lds_gbs=lds / t_s / 1e9,
            u_ta=v["ta_busy"] / (v["grbm"] * 80) if v["grbm"] else 0,
            u_valu=v["valu_slots"] / (v["grbm"] * 80) if v["grbm"] else 0,
            u_mmop=v["mmop_active"] / (v["grbm"] * 80) if v["grbm"] else 0,
            u_lds=v["lds_idx"] / (v["grbm"] * 80) if v["grbm"] else 0,
            u_ta_act=v["ta_busy"] / act_n, u_valu_act=v["valu_slots"] / act_n,
            u_mmop_act=v["mmop_active"] / act_n, u_lds_act=v["lds_idx"] / act_n,
            conf_i=v["lds_conf"] / v["lds_insts"] if v["lds_insts"] else 0,
            wait_i=v["lds_wait"] / v["lds_insts"] if v["lds_insts"] else 0,
            f_mem=f_mem * 1e6, f_cmp=f_cmp * 1e6, f_ta=f_ta * 1e6, f_val=f_val * 1e6,
            f_lds=f_lds * 1e6, f_chain=f_chain * 1e6, floor_us=floor * 1e6,
            binding=binding, gap=gap, req_bytes=req,
            w_gap=(hbm / req if req else None),
            resident=resident, rounds=rounds)
    return dict(T=T, H=H, NT=NT, cards=cards, launchers=launchers)


def main(shapes=None):
    shapes = shapes or sorted(p.name for p in RAW.iterdir()
                              if p.is_dir() and p.name[0].isdigit())
    out = {}
    for s in shapes:
        c = card(s)
        classify(c)
        out[s] = c
        tot = sum(v["t_us"] for v in c["cards"].values())
        fl = sum(v["floor_us"] for v in c["cards"].values())
        print(f"\n### {s}: kernels {len(c['cards'])}  total {tot/1e3:.1f} ms  sum-floor {fl/1e3:.1f} ms")
        for k, v in sorted(c["cards"].items(), key=lambda x: -x[1]["t_us"]):
            print(f"{k:26s} t={v['t_us']:8.1f} floor={v['floor_us']:8.1f} ({v['binding']:7s}) "
                  f"gap={v['gap']:5.2f}x  ta={v['u_ta']:.2f} valu={v['u_valu']:.2f} "
                  f"mmop={v['u_mmop']:.2f} hbm={v['hbm_gbs']:6.0f}GB/s req={v['req_bytes'] and v['req_bytes']/1e6 or 0:7.0f}MB"
                  f" w={v['w_gap'] and round(v['w_gap'],2)}")
    json.dump(out, open(RES / "cards.json", "w"), indent=1)




# ------------------------------------------------------------- gap classification (task sec.5)
def classify(c: dict) -> None:
    """Add per-kernel: primary/secondary constraint, at_ceiling flag, gap types U/W/C/S."""
    for key, v in c["cards"].items():
        act = min(80, v["ctas"])
        utils = {
            "HBM": (v["hbm_rd"] + v["hbm_wr"]) / max(1e-9, v["t_us"] * 1e-6) / HBW_R,
            "L2": v["tcc_req"] * 64 / max(1e-9, v["t_us"] * 1e-6) / 6100e9 if v["tcc_req"] else 0,
            "LDS": v["lds_idx"] / max(1, v["grbm"] * act) if v["grbm"] else 0,
            "TA": v["ta_busy"] / max(1, v["grbm"] * act) if v["grbm"] else 0,
            "VALU": v["valu_slots"] / max(1, v["grbm"] * act) if v["grbm"] else 0,
            "MMAC": v["mmop_active"] / max(1, v["grbm"] * act) if v["grbm"] else 0,
        }
        chain_u = 0.0
        if key in RECURRENCE:
            w_o = RECURRENCE[key] == "with_o"
            l = LSTEP["1cu"] if v["ctas"] <= 84 else LSTEP["2cu"]
            chain_u = c["NT"] * (l[0] if w_o else l[1]) / max(1e-9, v["t_us"] * 1e-6)
            utils["chain"] = chain_u
        order = sorted(utils.items(), key=lambda x: -x[1])
        primary, secondary = order[0], order[1]
        # work ratio on the binding resource
        wr = None
        if primary[0] == "HBM" and v.get("req_bytes"):
            wr = v["w_gap"]
        elif primary[0] in ("TA", "L2") and v.get("req_bytes"):
            vreq = v["req_bytes"] / 1024                    # best-pattern insts
            wr = v["vmem"] / vreq if vreq else None
        elif primary[0] == "MMAC":
            wr = 1.0        # SQ_INSTS_MMOP == ISA loop x trips (verify.md: dev=0)
        elif primary[0] == "VALU":
            wr = v["valu_insts"] and (v["valu_slots"] / max(1, v["valu_insts"]))
        elif primary[0] == "LDS":
            wr = 1 + v["conf_i"]                            # replay amplification proxy
        # gap type classification
        gaps = []
        if chain_u and (chain_u < 0.999 or v["ctas"] < 80):
            gaps.append("C")
        if wr and wr > 1.2:
            gaps.append("W")
        if primary[1] < 0.6:
            gaps.append("U")
        if v.get("req_bytes") and (v["hbm_rd"] + v["hbm_wr"]) / v["req_bytes"] > 1.2:
            gaps.append("W")
        # S: interface-dominated kernel (most measured HBM is inter-kernel tensor IO)
        if v.get("req_bytes") and v["flops"] < 1e10 and (v["hbm_rd"] + v["hbm_wr"]) > 0:
            gaps.append("S")
        at = (primary[1] >= 0.85) and (wr is None or wr <= 1.2)
        if key in RECURRENCE:
            at = chain_u >= 0.87 and v["ctas"] >= 80      # per-step within 15% of t_step_min
        v.update(dict(primary=primary[0], u_primary=primary[1],
                      secondary=secondary[0], u_secondary=secondary[1],
                      work_ratio=wr, at_ceiling=at, gap_types="".join(gaps) or "-"))

if __name__ == "__main__":
    main(sys.argv[1:] or None)
