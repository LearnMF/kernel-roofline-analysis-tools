"""Bytes vs bandwidth, per functional stage: FLA's DRAM bytes (ncu on A800, --cache-control none, one fwd+bwd) and
our HBM bytes (BW ledger, pmc), against the device times of FLA on H20 and ours on BW1000.  Shows whether the H20
gap is "we move more bytes" or "H20 moves bytes faster".
    python3 h20_bytes.py <results root> <ledger dir with T_H/ledger.json + pmc csv> <T> <H>"""
import collections
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
root, ldir, T, H = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
X3, XH = root / "kda-xplat-3way-20261017", root / "kda-xplat-h20-20261012"
_src = open(Path(__file__).resolve().parent / "h20_gap.py").read()
_i = _src.index("STAGES = [")
_g = {}
exec(_src[_i:_src.index("]\n", _src.index('("tail (cumsum', _i)) + 2], _g)
STAGES = _g["STAGES"]

# FLA bytes per kernel occurrence (ncu ID order)
rows = [r for r in csv.reader(open(X3 / "raw" / f"a800_ncu_fla_bytes_{T}_{H}.csv")) if len(r) > 14 and r[0].isdigit()]
by_id = collections.OrderedDict()
for r in rows:
    by_id.setdefault(int(r[0]), {"name": r[4]})[r[12]] = float(r[14].replace(",", ""))
seen, fla = collections.Counter(), {}
for d in by_id.values():
    seen[d["name"]] += 1
    fla[f"{d['name']}#{seen[d['name']]}"] = (d.get("dram__bytes_read.sum", 0) + d.get("dram__bytes_write.sum", 0),
                                              d.get("gpu__time_duration.sum", 0) / 1e3)
# FLA times on H20, ours times (rva kb) and ours bytes (pmc via byte_ledger's per-kernel parse)
h20 = {}
for line in open(XH / "raw" / "h20_kb" / f"kb_{T}_{H}.json"):
    if line.strip():
        d = json.loads(line)
        if d.get("arm") == "fla":
            h20 = {k: v * 1000.0 for k, v in d["kernels"]}
ours_t = json.load(open(X3 / "raw" / "bw_kb_now.json"))[f"g2_{T}_{H}"]
led = json.load(open(ldir / f"{T}_{H}" / "ledger.json"))
hbm = {k: v["rd"] + v["wr"] for k, v in led["hbm"].items()}
LAUNCH = [["fwd_prep"], ["fwd_h_o"], ["head_gate", "l2n_apply", "wu", "fwd_h_bn"], ["dav"], ["dhu_bn"], ["wy_s4_split"],
          ["bwd_tail"]]
print(f"### {T // 1024}K x {H}: bytes (MB) and achieved bandwidth (GB/s) per stage\n")
print("| stage | FLA bytes (A800 ncu) | ours bytes (BW pmc) | ours / FLA bytes | FLA on H20 us | FLA H20 GB/s | ours us | ours GB/s |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
ug, ua = set(), set()
tot = [0, 0, 0, 0]
for (name, gk, fk), lk in zip(STAGES, LAUNCH):
    fb = sum(fla.get(k, (0, 0))[0] for k in fk) / 1e6
    ht = sum(h20.get(k, 0) for k in fk)
    ob = sum(hbm.get(k, 0) for k in lk) / 1e6
    ot = sum(v for k, v in ours_t.items() if k in gk)
    ug |= set(gk)
    ua |= set(fk)
    tot = [tot[0] + fb, tot[1] + ob, tot[2] + ht, tot[3] + ot]
    print(f"| {name} | {fb:.0f} | {ob:.0f} | {ob / fb:.2f} | {ht:.0f} | {fb / ht * 1e3:.0f} | {ot:.0f} | {ob / ot * 1e3:.0f} |")
fo = sum(v[0] for k, v in fla.items() if k not in ua) / 1e6
ho = sum(v for k, v in h20.items() if k not in ua)
print(f"| other | {fo:.0f} | -- | -- | {ho:.0f} | -- | -- | -- |")
print(f"| **total** | **{tot[0] + fo:.0f}** | **{tot[1]:.0f}** | **{tot[1] / (tot[0] + fo):.2f}** | **{tot[2] + ho:.0f}** | "
      f"**{(tot[0] + fo) / (tot[2] + ho) * 1e3:.0f}** | **{tot[3]:.0f}** | **{tot[1] / tot[3] * 1e3:.0f}** |")
