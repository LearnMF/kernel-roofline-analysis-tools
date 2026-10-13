"""Instruction mix of a kernel's hottest loop from a `hipcc --cuda-device-only -S` listing.

The loop is the largest backward branch region inside the kernel (a branch to a label that appears
earlier in the same function).  Prints the instruction count by class and by opcode.

    python3 isa_loop_mix.py <file.s> <kernel-symbol-substring> [<file2.s> ...] [--top 30]
"""
import argparse
import collections
import re

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--kernel", required=True, help="substring of the mangled kernel symbol")
ap.add_argument("--top", type=int, default=0)
ap.add_argument("--whole", action="store_true", help="count the whole kernel, not only the hottest loop")
a = ap.parse_args()

CLASSES = [
    ("mmac", r"^v_mmac"), ("vmem_load", r"^(global|buffer|flat)_load"), ("vmem_store", r"^(global|buffer|flat)_store"),
    ("lds_read", r"^ds_read"), ("lds_write", r"^ds_write"), ("lds_other", r"^ds_"),
    ("waitcnt", r"^s_waitcnt"), ("barrier", r"^s_barrier"), ("salu", r"^s_"),
    ("cvt_bf16_rne", r"^v_(bfe_u32|add3_u32)"), ("addr64", r"^v_(add|sub)c?_co_u32|^v_addc_co|^v_subb|^v_lshl.*_b64|^v_lshlrev_b64"),
    ("trans", r"^v_(exp|log|rcp|rsq|sqrt|sin|cos)_"), ("valu", r"^v_"),
]


def loop_body(path, ksub, whole=False):
    lines = open(path).read().split("\n")
    start = next(i for i, ln in enumerate(lines) if re.match(r"^\S+:", ln) and ksub in ln and not ln.startswith("."))
    end = next(i for i in range(start + 1, len(lines)) if lines[i].strip().startswith("s_endpgm"))
    labels = {}
    best = None
    for i in range(start, end + 1):
        m = re.match(r"^(\.LBB\S+):", lines[i])
        if m:
            labels[m.group(1)] = i
        m = re.search(r"s_cbranch_\w+\s+(\.LBB\S+)", lines[i]) or re.search(r"s_branch\s+(\.LBB\S+)", lines[i])
        if m and m.group(1) in labels:
            span = (labels[m.group(1)], i)
            if best is None or span[1] - span[0] > best[1] - best[0]:
                best = span
    if whole:
        best = (start, end)
    assert best, f"no loop in {ksub}"
    ins = []
    for ln in lines[best[0]:best[1] + 1]:
        s = ln.strip()
        if not s or s.startswith((";", ".")) or s.endswith(":"):
            continue
        ins.append(s.split()[0])
    return ins


for f in a.files:
    ins = loop_body(f, a.kernel, a.whole)
    cls = collections.Counter()
    for op in ins:
        for name, rx in CLASSES:
            if re.match(rx, op):
                cls[name] += 1
                break
    print(f"== {f}  [{a.kernel}]  {'kernel' if a.whole else 'loop'} instructions: {len(ins)}")
    print("   " + "  ".join(f"{k}={cls[k]}" for k, _ in CLASSES if cls[k]))
    if a.top:
        for op, n in collections.Counter(ins).most_common(a.top):
            print(f"   {n:4d} {op}")
