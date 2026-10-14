"""Global layout assignment, first pass: price layout candidates of the s4b-side interface tensors with the
constructive model (trace_sim + per-instruction TA cost from the access pattern, kda_layout / vmem_bench L2 table).

Per kernel, the dynamic VMEM instructions of the production trace are mapped to their source access (order in the
trace = order in the source; see the s4b grammar below) and given the TA occupancy of their access pattern; a layout
candidate changes the patterns of the producer, of s4b and of the other consumers.  Kernel time = trace_sim on the
busiest CU.  The HBM part (hbm_frac) is unchanged (same bytes).
    python3 layout_eval.py [--shape 16384_12]"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import trace_sim as T  # noqa: E402
import derive11 as D  # noqa: E402

# L2-resident TA cycles per wave instruction by access pattern (vmem_bench, raw/vmem_bench.txt)
PAT = {"ld16x64": 43.0, "ld16x32": 41.0, "ldg4x64": 14.0, "ldc1k": 30.0, "ldc512": 21.0, "ld4x128": 23.0,
       "ldrow1": 13.0, "ld2x512": 24.0, "ld2x256": 23.0, "ldc256": 12.0, "ldg4x64s": 13.0,
       "st16x64": 52.0, "st16x32": 51.0, "stc1k": 59.0, "st4x256": 57.5, "stc256": 15.0, "stc512": 29.0}


def s4b_map(wave):
    """pattern label per VMEM instruction of an s4b wave (grammar from the trace: prologue | kt steps)"""
    v = [a.split()[0] for a in wave if a.startswith("global_")]
    lab = []
    first_kt = None
    # kt steps start after the 4 h/dh staging loads that precede the first s_barrier-delimited 'x XXXX xx' group;
    # locate them as the first dwordx2 that is followed by 4 dwordx4 then 2 dwordx2
    for i in range(len(v) - 7):
        if (v[i] == "global_load_dwordx2" and all(x == "global_load_dwordx4" for x in v[i + 1:i + 5])
                and v[i + 5] == v[i + 6] == "global_load_dwordx2" and "store" in v[i + 7]):
            first_kt = i
            break
    nx2 = 0
    for i, op in enumerate(v):
        if first_kt is not None and i >= first_kt:
            break
        if op == "global_load_dwordx2":
            lab.append("ld16x32" if nx2 < 12 else "ldc512")      # aa (4) + ado (8): row-major frags; avn: natC
            nx2 += 1
        elif op == "global_load_dword":
            ng = sum(1 for x in lab if x and x.startswith("ldg4x64"))
            lab.append("ldg4x64:dA" if ng < 10 else "ldg4x64:dAqk")
        elif op == "global_load_dwordx4":
            lab.append(None)
        else:
            lab.append(None)
    # prologue dwordx4: the last 4 before the kt loop are the h/dh staging (natives), the others dak / aqA rows
    idx4 = [k for k, op in enumerate(v[:first_kt]) if op == "global_load_dwordx4"]
    for n, k in enumerate(idx4):
        lab[k] = "ldc1k" if n >= len(idx4) - 4 else "ld16x64:dA" if n < 4 else "ld16x64:dAqk"
    k = first_kt
    while k is not None and k < len(v):
        op = v[k]
        if op == "global_load_dwordx2":
            # x (dwT) then later xx (kk, qq)
            lab.append("ld4x128" if (k + 1 < len(v) and v[k + 1] == "global_load_dwordx4") else "ld16x32:kq")
        elif op == "global_load_dwordx4":
            # within XXXX: gv, gn, h, dh (the last two steps carry only gv, gn)
            j = 0
            while k - j - 1 >= 0 and v[k - j - 1] == "global_load_dwordx4":
                j += 1
            lab.append(["ld16x64:g", "ldrow1", "ldc1k", "ldc1k"][min(j, 3)])
        elif op.startswith("global_store_dwordx4"):
            lab.append("st16x64")
        else:
            lab.append(None)
        k += 1
    return lab


def width_label(op, kernel):
    """pattern labels of the other kernels' accesses by opcode (their VMEM classes are unambiguous)"""
    if kernel == "gate_kernel" and op.startswith("global_store"):
        return "stc256:g"
    if kernel == "g2_wu_kernel" and op == "global_load_dwordx4":
        return "ld2x512:g"
    if kernel == "g2_wu_kernel" and op == "global_load_dwordx2":
        return "ld2x256:kq"
    if kernel == "g2_dav_kernel" and op == "global_store_dwordx4":
        return "st4x256:dAqk"
    if kernel == "wy_s4a" and op == "global_store_dwordx4":
        return "st16x64:dA"
    if kernel == "l2n_apply" and op.startswith("global_store"):
        return "stc512:kq"
    return None


# HBM-sourced cost per wave instruction at full concurrency (vmem_bench HBM column, scatter, pitch 3-12 KB): the TA is
# held for the DRAM transfer, so the cost reflects the pattern's DRAM efficiency (README section 13)
PAT_HBM = {"ld16x64": 206.0, "ld16x32": 173.0, "ldg4x64": 53.0, "ldc1k": 99.5, "ldc512": 48.8, "ld4x128": 69.0,
           "ldrow1": 126.0, "ld2x512": 113.0, "ld2x256": 48.6, "ldc256": 28.9,
           "ldg4x64s": 40.0,   # 4 rows x 64 B at a 256 B pitch inside one 1 KB block: NOT MEASURED, between the
                               # contiguous 256 B (28.9) and the 3 KB-pitch 4 x 64 B (52.3) points
           "st16x64": 230.0, "st16x32": 210.0, "stc1k": 110.8, "st4x256": 130.0, "stc256": 31.8, "stc512": 54.2}
MIX = {"on": True}
# s4b: HBM fraction per access class (64 B DRAM sectors; reuse across kt steps / waves); bytes-weighted mean 0.79 vs
# the PMC hbm_frac 0.82 of the kernel
S4B_F = {"ld16x64:g": 1.0, "ld16x32:kq": 0.5, "ld16x64:dA": 1.0, "ld16x64:dAqk": 1.0, "ldg4x64": 0.25,
         "ldg4x64:dA": 0.25, "ldg4x64:dAqk": 0.25, "ld16x32": 0.5, "ldc512": 1.0, "ldc1k": 1.0, "ldc1k:dA": 1.0,
         "ldc1k:dAqk": 1.0, "ld4x128": 1.0, "ldrow1": 0.05, "ldg4x64s:dA": 0.25, "ldg4x64s:dAqk": 0.25}


CANDIDATES = {   # tensor -> label substitutions when it is stored in the s4b-native fragment layout
    "baseline": {},
    "g native": {"ld16x64:g": "ldc1k", "stc256:g": "st16x32", "ld2x512:g": "ld16x64"},
    "kn/qn native": {"ld16x32:kq": "ldc512", "stc512:kq": "st16x32", "ld2x256:kq": "ld16x32"},
    "dA native (row reads)": {"ld16x64:dA": "ldc1k:dA", "st16x64:dA": "stc1k"},
    "dA native (rows + transposed reads)": {"ld16x64:dA": "ldc1k:dA", "st16x64:dA": "stc1k", "ldg4x64:dA": "ldg4x64s:dA"},
    "dAqk native (row reads)": {"ld16x64:dAqk": "ldc1k:dAqk", "st4x256:dAqk": "stc1k"},
    "dAqk native (rows + transposed reads)": {"ld16x64:dAqk": "ldc1k:dAqk", "st4x256:dAqk": "stc1k", "ldg4x64:dAqk": "ldg4x64s:dAqk"},
    "dA + dAqk native (all)": {"ld16x64:dA": "ldc1k:dA", "st16x64:dA": "stc1k", "ldg4x64:dA": "ldg4x64s:dA",
                               "ld16x64:dAqk": "ldc1k:dAqk", "st4x256:dAqk": "stc1k", "ldg4x64:dAqk": "ldg4x64s:dAqk"},
}


def occ_lists(kernel, waves, subst, f=0.0):
    out = []
    for w in waves:
        if kernel == "wy_s4b":
            labs = s4b_map(w)
        else:
            labs = [width_label(a.split()[0], kernel) for a in w if a.startswith("global_")]
        occ = []
        for lab in labs:
            if lab is None:
                occ.append(None)
                continue
            lab = subst.get(lab, lab)
            key = lab.split(":")[0]
            fx = 1.0 if key.startswith("st") else (S4B_F.get(lab, f) if kernel == "wy_s4b" else f)
            occ.append((1 - fx) * PAT[key] + fx * PAT_HBM[key] if MIX["on"] else PAT[key])
        out.append(occ)
    return out


def kernel_time(sh, card, name, subst):
    c = D.CARDS[sh]["cards"][card]
    tr = json.load(open(os.path.join(D.TRACES, name + ".json")))
    progs = [T.decode_prog(w) for w in tr["waves"]]
    hf = min(1.0, c["hbm_rd"] / max(1.0, T.loaded_bytes(progs) * c["ctas"]))
    occ = occ_lists(name, tr["waves"], subst, hf)
    # the mixed TA occupancy already holds the DRAM time of the HBM-sourced part -> no separate per-CU HBM queue
    return T.simulate(progs, D.residency(c), math.ceil(c["ctas"] / 80), 0.0 if MIX["on"] else hf, min(80, c["ctas"]),
                      ta_occ=occ, vmem_lat=T.S.LAT_L2 + hf * (T.S.LAT_HBM - T.S.LAT_L2)) / 1500.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="16384_12")
    a = ap.parse_args()
    ks = [("gate#1", "gate_kernel"), ("l2n_apply#1", "l2n_apply"), ("wu#1", "g2_wu_kernel"), ("dav#1", "g2_dav_kernel"),
          ("wy_s4a#1", "wy_s4a"), ("wy_s4b#1", "wy_s4b")]
    base = {n: kernel_time(a.shape, c, n, {}) for c, n in ks}
    meas = {n: D.CARDS[a.shape]["cards"][c]["t_us"] for c, n in ks}
    print(f"### {a.shape}: pattern-aware baseline vs measured (us): " +
          ", ".join(f"{n} {t:.0f}/{meas[n]:.0f}" for n, t in base.items()))
    print("\n| candidate | " + " | ".join(n for _, n in ks) + " | net us |")
    print("|---|" + "---:|" * (len(ks) + 1))
    for cand, sub in CANDIDATES.items():
        if cand == "baseline":
            continue
        d = {n: kernel_time(a.shape, c, n, sub) - base[n] for c, n in ks}
        print(f"| {cand} | " + " | ".join(f"{d[n]:+.1f}" for _, n in ks) + f" | **{sum(d.values()):+.1f}** |")


if __name__ == "__main__":
    main()
