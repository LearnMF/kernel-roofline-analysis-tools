"""KDA byte ledger: algorithm-required bytes vs launcher interface bytes vs measured HBM bytes, per launcher / kernel.

Inputs (one shape directory): optrace.json (kra LauncherCapture of one F+B), pmc_read.csv / pmc_write.csv (hipprof
--pmc-read / --pmc-write, a warmup + one measured iteration), trace_1.hipkernel.csv (hip-trace kernel averages).
    python3 byte_ledger.py <dir> --T 16384 --H 12 [--json out.json]
Measured HBM bytes per kernel (TCC EA requests, same formula as ceiling/parse_raw.py) are taken from the LAST
iteration (each kernel name's second half of occurrences).  Interface bytes (optrace) count each tensor passed to a
launcher once (inputs read, outputs written); scratch inside a launcher (prep_a->prep_b, s4a->s4b, tail partials) is
invisible there and shows up as measured - interface."""
import argparse
import collections
import csv
import json
import re

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--T", type=int, required=True)
ap.add_argument("--H", type=int, required=True)
ap.add_argument("--json", default=None)
a = ap.parse_args()
TH = a.T * a.H
MB = 1e6
# kda_bound P1 (C=64, d=128): algorithm-required HBM bytes per token*head for fwd+bwd
ALG_R, ALG_W = 2576, 1548

LMAP = [  # kernel-name regex -> launcher
    (r"g2_fwd_prep_a_kernel|g2_fwd_prep_b_kernel", "fwd_prep"),
    (r"g2_fwd_h_v8o4_kernelILb0ELi1E|g2_fwd_h_v8o4_kernel<false, 1,", "fwd_h_o"),
    (r"g2_fwd_h_v8o4_kernelILb0ELi4E|g2_fwd_h_v8o4_kernel<false, 4,|g2_gn_kp|g2_exp2", "fwd_h_bn"),
    (r"l2n", "l2n_apply"),
    (r"g2_gate_kernel|head_gate", "head_gate"),
    (r"g2_wu_kernel", "wu"),
    (r"g2_dav_kernel", "dav"),
    (r"g2_dhu_v8b|g2_dhu", "dhu_bn"),
    (r"p3_wy_s4a_kernel|p3_wy_s4b_kernel", "wy_s4_split"),
    (r"g2_bwd_tail", "bwd_tail"),
]


def launcher_of(name):
    for rx, l in LMAP:
        if re.search(rx, name):
            return l
    return None


def short(name):
    m = re.search(r"g2(?:..)?(\w+?kernel)", name) or re.search(r"(p3_\w+?kernel)", name)
    n = m.group(1) if m else name[:40]
    t = re.search(r"kernelIL?b?[01]?E?L?i?(\d)E", name)
    return n


def read_pmc(path, kind):
    rows = []
    with open(path) as f:
        r = csv.reader(f)
        hdr = next(r)
        for row in r:
            if len(row) < len(hdr):
                continue
            d = dict(zip(hdr, row))
            nm = d["KernelName"]

            def s(prefix):
                return sum(float(v) for k, v in d.items() if k.startswith(prefix + "[") and v not in ("", None))
            if kind == "rd":
                b = 64 * (s("TCC_EA_RDREQ") - s("TCC_EA_RDREQ_32B")) + 32 * s("TCC_EA_RDREQ_32B")
            else:
                b = 64 * s("TCC_EA_WRREQ_64B") + 32 * (s("TCC_EA_WRREQ") - s("TCC_EA_WRREQ_64B"))
            rows.append((nm, b))
    return rows


def last_iter(rows):
    by = collections.defaultdict(list)
    for nm, b in rows:
        by[nm].append(b)
    return {nm: sum(v[len(v) // 2:]) / max(1, len(v) - len(v) // 2) * 1 for nm, v in by.items()}, \
        {nm: len(v) - len(v) // 2 for nm, v in by.items()}


rd, ncall = last_iter(read_pmc(f"{a.dir}/pmc_read.csv", "rd"))
wr, _ = last_iter(read_pmc(f"{a.dir}/pmc_write.csv", "wr"))
times = {}
with open(f"{a.dir}/trace_1.hipkernel.csv") as f:
    for row in csv.DictReader(f):
        times[row["Name"]] = float(row["AverageNs"]) / 1e3
tr = json.load(open(f"{a.dir}/optrace.json"))
iface = collections.defaultdict(lambda: [0, 0, 0])
for c in tr["calls"]:
    x = iface[c["launcher"]]
    x[0] += c["in_bytes"]
    x[1] += c["out_bytes"]
    x[2] += 1

kern = []
for nm in rd:
    l = launcher_of(nm)
    if l is None:
        continue
    t = next((v for k, v in times.items() if k.replace("void ", "").split("(")[0] in nm or
              re.sub(r"\W", "", k)[:30] in re.sub(r"\W", "", nm)), None)
    kern.append(dict(name=nm, launcher=l, calls=ncall[nm], rd=rd[nm] * ncall[nm], wr=wr.get(nm, 0) * ncall[nm]))
lt = collections.defaultdict(lambda: dict(rd=0, wr=0, kernels=[]))
for k in kern:
    x = lt[k["launcher"]]
    x["rd"] += k["rd"]
    x["wr"] += k["wr"]
    x["kernels"].append(k)
# hip-trace kernel times grouped by launcher (names in hipkernel.csv are demangled)
ltime = collections.defaultdict(float)
DEM = [(r"fwd_prep_a_kernel|fwd_prep_b_kernel", "fwd_prep"), (r"fwd_h_v8o4_kernel<false, 1,", "fwd_h_o"),
       (r"fwd_h_v8o4_kernel<false, 4,|gn_kp|exp2", "fwd_h_bn"), (r"l2n", "l2n_apply"), (r"gate_kernel", "head_gate"),
       (r"wu_kernel", "wu"), (r"dav_kernel", "dav"), (r"dhu_", "dhu_bn"), (r"wy_s4a|wy_s4b", "wy_s4_split"),
       (r"bwd_tail", "bwd_tail")]
with open(f"{a.dir}/trace_1.hipkernel.csv") as f:
    for row in csv.DictReader(f):
        for rx, l in DEM:
            if re.search(rx, row["Name"]):
                ltime[l] += float(row["TotalDurationNs"]) / 1e3 / 4    # iters 3 + warmup 1
                break
order = ["fwd_prep", "fwd_h_o", "l2n_apply", "head_gate", "wu", "fwd_h_bn", "dav", "dhu_bn", "wy_s4_split", "bwd_tail"]
print(f"== {a.dir}  T={a.T} H={a.H}  (MB; B/th = bytes per token*head)")
print(f"{'launcher':12s} {'us':>7s} {'iface rd':>9s} {'iface wr':>9s} {'HBM rd':>8s} {'HBM wr':>8s} {'HBM/iface':>9s} {'HBM B/th':>9s}")
tot = dict(us=0, ir=0, iw=0, hr=0, hw=0)
for l in order:
    i = iface.get(l, [0, 0, 0])
    x = lt.get(l, dict(rd=0, wr=0))
    us = ltime.get(l, 0)
    hb = x["rd"] + x["wr"]
    ib = i[0] + i[1]
    tot["us"] += us
    tot["ir"] += i[0]
    tot["iw"] += i[1]
    tot["hr"] += x["rd"]
    tot["hw"] += x["wr"]
    print(f"{l:12s} {us:7.0f} {i[0] / MB:9.1f} {i[1] / MB:9.1f} {x['rd'] / MB:8.1f} {x['wr'] / MB:8.1f} "
          f"{hb / max(1, ib):9.2f} {hb / TH:9.0f}")
print(f"{'TOTAL':12s} {tot['us']:7.0f} {tot['ir'] / MB:9.1f} {tot['iw'] / MB:9.1f} {tot['hr'] / MB:8.1f} {tot['hw'] / MB:8.1f} "
      f"{(tot['hr'] + tot['hw']) / max(1, tot['ir'] + tot['iw']):9.2f} {(tot['hr'] + tot['hw']) / TH:9.0f}")
alg = (ALG_R + ALG_W) * TH
print(f"algorithm-required (kda_bound P1): {alg / MB:.1f} MB = {ALG_R + ALG_W} B/th ; measured HBM / required = "
      f"{(tot['hr'] + tot['hw']) / alg:.2f}x ; interface / required = {(tot['ir'] + tot['iw']) / alg:.2f}x")
print("\nper kernel (measured HBM):")
for k in sorted(kern, key=lambda k: -(k["rd"] + k["wr"])):
    print(f"  {k['launcher']:12s} {(k['rd'] + k['wr']) / MB:8.1f} MB (rd {k['rd'] / MB:7.1f} wr {k['wr'] / MB:7.1f}) x{k['calls']}  "
          f"{k['name'][:90]}")
if a.json:
    json.dump(dict(T=a.T, H=a.H, iface={k: v for k, v in iface.items()}, hbm={k: dict(rd=v["rd"], wr=v["wr"]) for k, v in lt.items()},
                   time_us=dict(ltime), alg_bytes=alg), open(a.json, "w"), indent=1)
