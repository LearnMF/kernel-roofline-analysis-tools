"""Per-kernel ISA statistics from the frozen tree's disassembly (g2_ops_hip.s).

For every kernel symbol it reports: MMAC instruction types present in the function
(and inside its hottest loop), the hottest backward-branch loop's instruction mix,
and whole-function totals.  Used by cards.py for exact MMAC FLOP typing and the
main-loop composition table.

    python3 isa_mix.py g2_ops_hip.s > isa_mix.json
"""
from __future__ import annotations

import collections
import json
import re
import sys

OPC = re.compile(r"^\s+(v_\w+|s_\w+|ds_\w+|global_\w+|flat_\w+|buffer_\w+|exp\w*)")
LABEL = re.compile(r"^(_ZN[\w$]+):")
BB = re.compile(r"^(\.LBB\d+_\d+):")
BRANCH = re.compile(r"^\s+s_cbranch_\w+\s+(\.LBB\d+_\d+)")


def _cat(op: str) -> str:
    if op.startswith("v_mmac") or op.startswith("v_smfmac"):
        return op
    if op in ("v_exp_f32", "v_exp_f16", "v_rcp_f32", "v_rcp_iflag_f32",
              "v_rsq_f32", "v_sqrt_f32", "v_log_f32", "v_sin_f32", "v_cos_f32"):
        return "transcendental"
    if op in ("v_bfe_u32", "v_bfe_i32", "v_add3_u32"):
        return "bf16_round"
    if op.startswith(("global_load", "flat_load", "buffer_load", "s_load")):
        return "vmem_rd" if not op.startswith("s_load") else "s_load"
    if op.startswith(("global_store", "flat_store", "buffer_store")):
        return "vmem_wr"
    if op.startswith("ds_read"):
        return "lds_rd_" + op.split("_")[-1]
    if op.startswith("ds_write"):
        return "lds_wr_" + op.split("_")[-1]
    if op == "s_waitcnt" or op.startswith("s_waitcnt"):
        return "waitcnt"
    if op == "s_barrier":
        return "barrier"
    if op in ("v_add_co_u32", "v_addc_co_u32", "v_sub_co_u32", "v_subb_co_u32",
              "v_add_co_ci_u32", "v_addc_co_ci_u32"):
        return "addr64"
    if op.startswith("s_"):
        return "scalar"
    return "valu"


def parse(path: str) -> dict:
    lines = open(path).read().splitlines()
    labels = [(i, LABEL.match(l).group(1)) for i, l in enumerate(lines) if LABEL.match(l)]
    labels.append((len(lines), "END"))
    out = {}
    for j, (i, name) in enumerate(labels[:-1]):
        body = lines[i:labels[j + 1][0]]
        ops = []
        bbs = {}
        for l in body:
            m = BB.match(l)
            if m:
                cur = m.group(1)
                continue
            m = OPC.match(l)
            if m:
                ops.append((m.group(1), cur if 'cur' in dir() else None))
            m2 = BRANCH.match(l)
            if m2 and 'cur' in dir():
                bbs.setdefault(cur, []).append(m2.group(1))
        cnt = collections.Counter(o for o, _ in ops)
        # hottest loop = backward branch with most instructions between target and branch
        loops = []
        lbl_idx = {m.group(1): k for k, l in enumerate(body) if (m := BB.match(l))}
        for k, l in enumerate(body):
            m = BRANCH.match(l)
            if not m:
                continue
            tgt = m.group(1)
            if tgt in lbl_idx and lbl_idx[tgt] < k:
                loops.append((lbl_idx[tgt], k))
        if loops:
            a, b = max(loops, key=lambda ab: ab[1] - ab[0])
            loop_ops = [OPC.match(l).group(1) for l in body[a:b + 1] if OPC.match(l)]
            lcnt = collections.Counter(loop_ops)
        else:
            lcnt = collections.Counter()
        def mix(c):
            g = collections.Counter(_cat(o) for o in c.elements())
            return dict(g)
        mm = {o: v for o, v in cnt.items() if "mmac" in o or "smfmac" in o}
        rec = dict(
            total_ops=sum(cnt.values()), mmac=mm, exp=cnt.get("v_exp_f32", 0),
            lds={o: v for o, v in cnt.items() if o.startswith("ds_")},
            vmem_rd=sum(v for o, v in cnt.items() if o.startswith(("global_load", "flat_load"))),
            vmem_wr=sum(v for o, v in cnt.items() if o.startswith(("global_store", "flat_store"))),
            barrier=cnt.get("s_barrier", 0), waitcnt=cnt.get("s_waitcnt", 0),
            loop_len=sum(lcnt.values()) if lcnt else 0,
            loop_mix=mix(lcnt), func_mix=mix(cnt))
        if rec["total_ops"] > out.get(name, {}).get("total_ops", -1):
            out[name] = rec
    return out


if __name__ == "__main__":
    d = parse(sys.argv[1])
    json.dump(d, open(sys.argv[2] if len(sys.argv) > 2 else "/dev/stdout", "w"), indent=1)
    for n, v in sorted(d.items(), key=lambda x: -x[1]["total_ops"])[:25]:
        if v["mmac"] or v["total_ops"] > 500:
            print(f"{n[:70]:70s} tot={v['total_ops']:6d} loop={v['loop_len']:5d} mmac={v['mmac']}")
