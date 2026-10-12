"""Stage-level comparison of ours (hip_kda G2 on BW) against FLA on an NVIDIA GPU (H20 / A800), same
functional stages as stage_compare.py (kra R15).
    python3 stage_compare_vs.py bw_kb.json nv_kb_dir_or_files... [--label H20] [--hw <attainable HBM ratio NV/BW>]"""
import glob
import json
import os
import sys

_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage_compare.py")).read()
_i = _src.index("STAGES = [")
_g = {}
exec(_src[_i:_src.index("]\n\n", _i) + 2], _g)      # the same stage map, without running the script
STAGES = _g["STAGES"]

args = [a for a in sys.argv[1:] if not a.startswith("--")]
label = sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else "NV"
HW = float(sys.argv[sys.argv.index("--hw") + 1]) if "--hw" in sys.argv else None
args = [a for a in args if a != label and (HW is None or a != str(HW))]
bw = json.load(open(args[0]))
nv = {}
for p in args[1:]:
    for f in (sorted(glob.glob(os.path.join(p, "*.json"))) if os.path.isdir(p) else [p]):
        for line in open(f):
            if line.strip():
                d = json.loads(line)
                if d.get("arm") == "fla":
                    nv[f"{d['T']}_{d['H']}"] = {k: v * 1000.0 for k, v in d["kernels"]}
pick = lambda d, names: sum(v for k, v in d.items() if k in names)
for key in sorted(nv, key=lambda s: tuple(map(int, s.split("_")))):
    g, a = bw.get(f"g2_{key}", {}), nv[key]
    if not g:
        continue
    T, H = key.split("_")
    hdr = f"| stage | ours (BW) | FLA ({label}) | {label} / ours |" + (f" {label} x hw / ours |" if HW else "")
    print(f"\n### {int(T) // 1024}K x {H} heads (TP{96 // int(H)}; device us; >1 = ours faster)\n\n{hdr}")
    print("|---|---:|---:|---:|" + ("---:|" if HW else ""))
    used_g, used_a, rows = set(), set(), []
    for name, gk, fk in STAGES:
        rows.append((name, pick(g, gk), pick(a, fk)))
        used_g |= set(gk); used_a |= set(fk)
    rows.append(("other (copies, fills, reductions)", sum(v for k, v in g.items() if k not in used_g),
                 sum(v for k, v in a.items() if k not in used_a)))
    for name, tg, ta in rows:
        r = f"{ta / tg:.2f}" if tg > 0 else "-"
        print(f"| {name} | {tg:.0f} | {ta:.0f} | {r} |" + (f" {ta * HW / tg:.2f} |" if HW and tg > 0 else (" - |" if HW else "")))
    TG, TA = sum(x[1] for x in rows), sum(x[2] for x in rows)
    print(f"| **total** | **{TG:.0f}** | **{TA:.0f}** | **{TA / TG:.2f}** |" + (f" **{TA * HW / TG:.2f}** |" if HW else ""))
