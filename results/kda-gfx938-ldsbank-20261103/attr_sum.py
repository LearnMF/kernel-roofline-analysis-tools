"""Per-phase LDS bank-conflict attribution from lb_attr.sh output (hipprof SQ_* sums, per CTA).
python3 attr_sum.py FILE.jsonl CTAS"""
import json, sys
DAV = ["pass-1 Vn staging (b16 st)", "VnA re-emit reads (b64)", "pass-1 MMAC B reads (b64)", "dA fp32 staging (b32 st)",
       "dA staging reads (b128)", "pass-2 do scatter (b16 st)", "dOT reads (b64)", "pass-2 MMAC B reads (b64)"]
WU = ["fill1 kbT (b16 st)", "w-MMAC B reads (b64)", "Wb tile (b16 st)", "Wb emit16 reads (b64)", "fill2 natural vb (b64 st)",
      "u-MMAC ds_read_m32x16", "HbN staging (b64 st)", "HbN ds_read_m32x16", "Qn tile (b16 st)", "Qn emit16 reads (b64)"]
rows = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
n = float(sys.argv[2])
by = {(r["tag"], r["k"]): r for r in rows}
for k, names in (("g2_dav", DAV), ("g2_wu", WU)):
    full = by.get(("full", k))
    if not full: continue
    print(f"== {k}: full conflicts {full['SQ_LDS_BANK_CONFLICT'] / n:.0f} / LDS active {full['SQ_LDS_IDX_ACTIVE'] / n:.0f} cycles per CTA")
    s = 0
    for i, nm in enumerate(names):
        r = by.get((f"only{i}", k))
        if not r: continue
        c = r["SQ_LDS_BANK_CONFLICT"] / n; s += c
        print(f"  {k[3:]}{i} {nm:30s} conflicts {c:6.0f}  insts {r['SQ_INSTS_LDS'] / n:5.0f}  active {r['SQ_LDS_IDX_ACTIVE'] / n:6.0f}")
    print(f"  sum of phases {s:.0f}")
