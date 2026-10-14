import re, sys, collections
# per-kernel cross-lane instruction inventory from hipcc -S output
cur, cnt = None, collections.defaultdict(collections.Counter)
for line in open(sys.argv[1]):
    m = re.match(r"^([A-Za-z_][\w.$]*):", line)
    if m:
        cur = m.group(1); continue
    if cur is None:
        continue
    s = line.strip()
    if not s or s.startswith((".", ";")):
        if s.startswith(".Lfunc_end"): cur = None
        continue
    op = s.split()[0]
    c = cnt[cur]
    c["tot"] += 1
    if op.startswith("ds_bpermute") or op.startswith("ds_permute"): c["bperm"] += 1
    elif op.startswith("ds_swizzle"): c["swz"] += 1
    elif "_dpp" in op or "dpp" in s.split(" ", 1)[-1] and "row_" in s: c["dpp"] += 1
    elif op.startswith(("v_readlane", "v_readfirstlane", "v_writelane")): c["rdlane"] += 1
    elif op.startswith("v_mmac"): c["mmac"] += 1
    elif op.startswith("v_") and ("quad_perm" in s or "row_shr" in s or "row_ror" in s or "row_bcast" in s or "row_mirror" in s): c["dpp"] += 1
for k, c in sorted(cnt.items(), key=lambda kv: -(kv[1]["bperm"] + kv[1]["dpp"] + kv[1]["swz"])):
    if c["bperm"] + c["dpp"] + c["swz"] == 0: continue
    print(f"{k[:90]:90s} tot={c['tot']:5d} bperm={c['bperm']:4d} dpp={c['dpp']:4d} swz={c['swz']:3d} rdlane={c['rdlane']:4d} mmac={c['mmac']:4d}")
