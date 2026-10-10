"""Stage-level competitor comparison (global-direction check, kra playbook R15):
the same functional stages of KDA fwd+bwd, timed for ours (hip_kda G2 on BW), FLA on the SAME BW
hardware, and FLA on A800 -- so that a deficit can be attributed to the implementation (FLA@BW
beats us on the same hardware = known achievable) or to the hardware (only FLA@A800 is faster).

    python3 stage_compare.py bw_breakdown.json nv_breakdown_occ.jsonl [--hw 1.32]

bw_breakdown.json: {"g2_T_H": {"kernel#k": us}, "fla_T_H": {...}} (hipprof, occurrence-keyed);
nv_breakdown_occ.jsonl: nv_kernel_breakdown.py output (CUPTI, occurrence-keyed, ms).
--hw: attainable-HBM ratio A800/BW (1771/1339) for the hardware-normalized column."""
import json
import sys

bw = json.load(open(sys.argv[1]))
nv = {}
for line in open(sys.argv[2]):
    d = json.loads(line)
    if d["arm"] == "fla":
        nv[f"{d['T']}_{d['H']}"] = {k: v * 1000.0 for k, v in d["kernels"]}
HW = float(sys.argv[sys.argv.index("--hw") + 1]) if "--hw" in sys.argv else 1771.2 / 1339.2

# functional stages: (name, ours kernels, FLA kernels).  FLA default recomputes gate cumsum,
# w/u and h in the backward (#2 occurrences); ours recomputes l2n, w/u (wu) and h (v8o4 #2).
STAGES = [
    ("forward: prep (l2norm, gate, intra A, w/u)",
     ["g2_gate_kernel#1", "g2_fwd_prep_a_kernel#1", "g2_fwd_prep_b_kernel#1"],
     ["l2norm_fwd_kernel#1", "l2norm_fwd_kernel#2", "kda_gate_chunk_cumsum_vector_kernel#1",
      "chunk_kda_fwd_kernel_intra_sub_chunk#1", "chunk_kda_fwd_kernel_inter_solve_fused#1",
      "recompute_w_u_fwd_kda_kernel#1"]),
    ("forward: recurrence h + output o", ["g2_fwd_h_v8o4_kernel#1"],
     ["chunk_gated_delta_rule_fwd_kernel_h_blockdim64#1", "chunk_gla_fwd_kernel_o#1"]),
    ("bwd recompute: q/k norm, gate, w/u, h", ["g2_l2n_apply_kernel#1", "g2_wu_kernel#1", "g2_fwd_h_v8o4_kernel#2"],
     ["kda_gate_chunk_cumsum_vector_kernel#2", "recompute_w_u_fwd_kda_kernel#2",
      "chunk_gated_delta_rule_fwd_kernel_h_blockdim64#2"]),
    ("bwd: dA_qk + dv (dAv)", ["g2_dav_kernel#1"], ["chunk_kda_bwd_kernel_dAv#1"]),
    ("bwd: state gradient dh (dhu)", ["g2_dhu_v8b_kernel#1", "g2_dhu_v8b4_kernel#1"],
     ["chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64#1"]),
    ("bwd: dq/dk/dg/dbeta/dA_kk (wy + intra)", ["p3_wy_s4a_kernel#1", "p3_wy_s4b_kernel#1"],
     ["chunk_kda_bwd_kernel_wy_dqkg_fused#1", "chunk_kda_bwd_kernel_intra#1"]),
    ("bwd tail: cumsum, gate bwd, l2norm bwd", ["g2_bwd_tail_kernel#1", "g2_bwd_tail_reduce_kernel#1"],
     ["chunk_local_cumsum_vector_kernel#1", "kda_gate_bwd_kernel#1", "l2norm_bwd_kernel#1", "l2norm_bwd_kernel#2"]),
]


def pick(d, names):
    return sum(v for k, v in d.items() if k in names)


for key in sorted(nv, key=lambda s: tuple(map(int, s.split("_")))):
    g, f, a = bw.get(f"g2_{key}", {}), bw.get(f"fla_{key}", {}), nv[key]
    if not g or not f:
        continue
    used_g, used_f, used_a = set(), set(), set()
    T, H = key.split("_")
    print(f"\n### {int(T) // 1024}K x {H} heads (device us; ratio = other / ours, >1 = ours faster)\n")
    print("| stage | ours (BW) | FLA (BW) | FLA (A800) | FLA@BW / ours | FLA@A800 / ours | FLA@A800 x hw / ours |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    rows = []
    for name, gk, fk in STAGES:
        tg, tf, ta = pick(g, gk), pick(f, fk), pick(a, fk)
        used_g |= set(gk); used_f |= set(fk); used_a |= set(fk)
        rows.append((name, tg, tf, ta))
    tg = sum(v for k, v in g.items() if k not in used_g)
    tf = sum(v for k, v in f.items() if k not in used_f)
    ta = sum(v for k, v in a.items() if k not in used_a)
    rows.append(("other (copies, fills, reductions)", tg, tf, ta))
    for name, tg, tf, ta in rows:
        r = lambda x: f"{x / tg:.2f}" if tg > 0 else "-"
        print(f"| {name} | {tg:.0f} | {tf:.0f} | {ta:.0f} | {r(tf)} | {r(ta)} | {r(ta * HW)} |")
    TG, TF, TA = (sum(x[i] for x in rows) for i in (1, 2, 3))
    print(f"| **total** | **{TG:.0f}** | **{TF:.0f}** | **{TA:.0f}** | **{TF / TG:.2f}** | **{TA / TG:.2f}** | **{TA * HW / TG:.2f}** |")
