"""Where the H20 gap is and how it moved: (1) H20 FLA default / ours (device time) at every BW milestone with data,
(2) stage-level ours (2026-10-12 and now) vs FLA on H20, with the absolute deficit per stage.
    python3 h20_gap.py <results root>"""
import json
import math
import statistics as st
import sys
from pathlib import Path

root = Path(sys.argv[1])
X8, XH, X3 = root / "kda-xplat-a800-20261010", root / "kda-xplat-h20-20261012", root / "kda-xplat-3way-20261017"


def nv(p):
    out = {}
    for line in open(p):
        if line.strip():
            d = json.loads(line)["data"]
            out.setdefault((d["T"], d["H"]), []).append(d["res"]["fla"]["kernel_fb_ms"])
    return {k: st.median(v) for k, v in out.items()}


def bwdev(p):
    return {tuple(map(int, k.split("x"))): v["g2"]["device_fb_ms"] for k, v in json.load(open(p)).items() if "g2" in v}


h20 = nv(XH / "raw" / "h20.jsonl")
miles = [("2026-10-10 first cross-platform run", bwdev(X8 / "raw" / "seq_bw_dev.json")),
         ("2026-10-12 first H20 run (a81ded8)", bwdev(XH / "raw" / "bw_dev_now.json")),
         ("2026-10-16 session start (P-65)", bwdev(X3 / "raw" / "bw_dev_session_start_merged.json")),
         ("2026-10-17 now (P-87, 931f9e3)", bwdev(X3 / "raw" / "bw_dev_now_merged.json"))]
shapes = sorted(h20)
TP = {12: "TP8", 24: "TP4", 48: "TP2", 96: "TP1"}
print("### H20 FLA default / ours (device time; > 1 = ours faster)\n")
print("| T | TP | H20 FLA ms | " + " | ".join(m[0] for m in miles) + " |")
print("|---|---|---:|" + "---:|" * len(miles))
for s in shapes:
    print(f"| {s[0] // 1024}K | {TP[s[1]]} | {h20[s]:.2f} | " + " | ".join(
        f"{h20[s] / m[1][s]:.2f} ({m[1][s]:.2f})" if s in m[1] else "--" for m in miles) + " |")


def gm(v):
    return math.exp(sum(map(math.log, v)) / len(v))


for name, sel in (("all 16", shapes), ("9 priority (16K-64K, TP>1)", [s for s in shapes if s[0] >= 16384 and s[1] <= 48]),
                  ("6 shapes of 10-10", [s for s in shapes if s in miles[0][1]])):
    cells = []
    for m in miles:
        if all(s in m[1] for s in sel):
            cells.append(f"{gm([h20[s] / m[1][s] for s in sel]):.3f}")
        else:
            cells.append("--")
    print(f"\ngeomean {name}: " + ", ".join(f"{m[0]}: {c}" for m, c in zip(miles, cells)))

STAGES = [
    ("forward prep (l2norm, gate, intra A, w/u)", ["g2_fwd_prep_a_kernel#1", "g2_fwd_prep_b_kernel#1"],
     ["l2norm_fwd_kernel#1", "l2norm_fwd_kernel#2", "kda_gate_chunk_cumsum_vector_kernel#1",
      "chunk_kda_fwd_kernel_intra_sub_chunk#1", "chunk_kda_fwd_kernel_inter_solve_fused#1", "recompute_w_u_fwd_kda_kernel#1"]),
    ("forward recurrence h + o", ["g2_fwd_h_v8o4_kernel#1", "g2_fwd_h_v8o4w_kernel#1"],
     ["chunk_gated_delta_rule_fwd_kernel_h_blockdim64#1", "chunk_gla_fwd_kernel_o#1"]),
    ("bwd recompute (gate, w/u [+v_new], h)", ["g2_gate_kernel#1", "g2_l2n_apply_kernel#1", "g2_wu_kernel#1",
                                              "g2_fwd_h_v8o4_kernel#2", "g2_fwd_h_v8o4w_kernel#2", "g2_gn_kp_kernel#1", "g2_exp2_kernel#1"],
     ["kda_gate_chunk_cumsum_vector_kernel#2", "recompute_w_u_fwd_kda_kernel#2", "chunk_gated_delta_rule_fwd_kernel_h_blockdim64#2"]),
    ("dA_qk + dv (dAv)", ["g2_dav_kernel#1"], ["chunk_kda_bwd_kernel_dAv#1"]),
    ("state gradient dh (dhu)", ["g2_dhu_v8b_kernel#1", "g2_dhu_v8b4_kernel#1"], ["chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64#1"]),
    ("dq/dk/dg/dbeta/dA_kk (wy + intra)", ["p3_wy_s4a_kernel#1", "p3_wy_s4b_kernel#1"],
     ["chunk_kda_bwd_kernel_wy_dqkg_fused#1", "chunk_kda_bwd_kernel_intra#1"]),
    ("tail (cumsum, gate bwd, l2norm bwd)", ["g2_bwd_tail_kernel#1", "g2_bwd_tail_reduce_kernel#1"],
     ["chunk_local_cumsum_vector_kernel#1", "kda_gate_bwd_kernel#1", "l2norm_bwd_kernel#1", "l2norm_bwd_kernel#2"]),
]
old = json.load(open(XH / "raw" / "bw_kernel_breakdown.json"))
now = json.load(open(X3 / "raw" / "bw_kb_now.json"))
nvkb = {}
for f in sorted((XH / "raw" / "h20_kb").glob("*.json")):
    for line in open(f):
        if line.strip():
            d = json.loads(line)
            if d.get("arm") == "fla":
                nvkb[f"{d['T']}_{d['H']}"] = {k: v * 1000.0 for k, v in d["kernels"]}
for key in sorted(nvkb, key=lambda s: tuple(map(int, s.split("_")))):
    T, H = map(int, key.split("_"))
    o, n, a = old.get(f"g2_{key}", {}), now.get(f"g2_{key}", {}), nvkb[key]
    print(f"\n### {T // 1024}K x {H} heads ({TP[H]}; device us)\n")
    print("| stage | ours 10-12 | ours now | FLA on H20 | H20 / ours now | ours now - H20 |")
    print("|---|---:|---:|---:|---:|---:|")
    ug, ua, rows = set(), set(), []
    for name, gk, fk in STAGES:
        rows.append((name, sum(v for k, v in o.items() if k in gk), sum(v for k, v in n.items() if k in gk),
                     sum(v for k, v in a.items() if k in fk)))
        ug |= set(gk)
        ua |= set(fk)
    rows.append(("other (copies, fills, reductions)", sum(v for k, v in o.items() if k not in ug),
                 sum(v for k, v in n.items() if k not in ug), sum(v for k, v in a.items() if k not in ua)))
    for name, to, tn, ta in rows:
        print(f"| {name} | {to:.0f} | {tn:.0f} | {ta:.0f} | {ta / tn:.2f} | {tn - ta:+.0f} |")
    TO, TN, TA = (sum(x[i] for x in rows) for i in (1, 2, 3))
    print(f"| **total** | **{TO:.0f}** | **{TN:.0f}** | **{TA:.0f}** | **{TA / TN:.2f}** | **{TN - TA:+.0f}** |")
