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
       "ldrow1": 13.0, "ld2x512": 24.0, "ld2x256": 23.0, "ldc256": 12.0,
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
            lab.append("ldg4x64")
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


CANDIDATES = {   # tensor -> label substitutions when it is stored in the s4b-native fragment layout
    "baseline": {},
    "g native": {"ld16x64:g": "ldc1k", "stc256:g": "st16x32", "ld2x512:g": "ld16x64"},
    "kn/qn native": {"ld16x32:kq": "ldc512", "stc512:kq": "st16x32", "ld2x256:kq": "ld16x32"},
    "dA native (row reads)": {"ld16x64:dA": "ldc1k", "st16x64:dA": "stc1k"},
    "dAqk native (row reads)": {"ld16x64:dAqk": "ldc1k", "st4x256:dAqk": "stc1k"},
}


def occ_lists(kernel, waves, subst):
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
            occ.append(PAT[lab.split(":")[0]])
        out.append(occ)
    return out


def kernel_time(sh, card, name, subst):
    c = D.CARDS[sh]["cards"][card]
    tr = json.load(open(os.path.join(D.TRACES, name + ".json")))
    progs = [T.decode_prog(w) for w in tr["waves"]]
    hf = min(1.0, c["hbm_rd"] / max(1.0, T.loaded_bytes(progs) * c["ctas"]))
    occ = occ_lists(name, tr["waves"], subst)
    return T.simulate(progs, D.residency(c), math.ceil(c["ctas"] / 80), hf, min(80, c["ctas"]), ta_occ=occ) / 1500.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="16384_12")
    a = ap.parse_args()
    ks = [("gate#1", "gate_kernel"), ("l2n_apply#1", "l2n_apply"), ("wu#1", "g2_wu_kernel"), ("dav#1", "g2_dav_kernel"),
          ("wy_s4a#1", "wy_s4a"), ("wy_s4b#1", "wy_s4b")]
    base = {n: kernel_time(a.shape, c, n, {}) for c, n in ks}
    print(f"### {a.shape}: pattern-aware baseline (us): " + ", ".join(f"{n} {t:.0f}" for n, t in base.items()))
    print("\n| candidate | " + " | ".join(n for _, n in ks) + " | net us |")
    print("|---|" + "---:|" * (len(ks) + 1))
    for cand, sub in CANDIDATES.items():
        if cand == "baseline":
            continue
        d = {n: kernel_time(a.shape, c, n, sub) - base[n] for c, n in ks}
        print(f"| {cand} | " + " | ".join(f"{d[n]:+.1f}" for _, n in ks) + f" | **{sum(d.values()):+.1f}** |")


if __name__ == "__main__":
    main()
