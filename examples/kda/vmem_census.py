"""Per-kernel VMEM census from a -g device ISA: every global load/store with its width, the source line of its
.loc, and whether it sits inside a backward-branch loop -- the input for access-pattern work (R54: TA cost follows
bytes and pattern).  Lines refer to the build's *_hip.cuh copies.
    python3 vmem_census.py <isa_g.s> <kernel-symbol-fragment> [--src <dir with *_hip.cuh>]"""
import argparse
import collections
import re

ap = argparse.ArgumentParser()
ap.add_argument("isa")
ap.add_argument("kernel")
ap.add_argument("--src", default=None)
a = ap.parse_args()
FILE = re.compile(r'^\s*\.file\s+(\d+)\s+"([^"]*)"(?:\s+"([^"]*)")?')
LOC = re.compile(r"^\s*\.loc\s+(\d+)\s+(\d+)")
BB = re.compile(r"^(\.LBB\d+_\d+):")
BR = re.compile(r"^\s+s_cbranch_\w+\s+(\.LBB\d+_\d+)|^\s+s_branch\s+(\.LBB\d+_\d+)")
VM = re.compile(r"^\s+(global|buffer|flat)_(load|store)_(\w+)")
files, body, cur, started, labels = {}, [], None, False, {}
for ln in open(a.isa, errors="replace"):
    fm = FILE.match(ln)
    if fm:
        files[fm.group(1)] = (fm.group(3) or fm.group(2)).split("/")[-1]
        continue
    if not started:
        head = ln.split(";")[0].rstrip()
        if not ln.startswith((" ", "\t", ".")) and head.endswith(":") and a.kernel in head:
            started = True
        continue
    if ln.startswith(".Lfunc_end") or (ln.startswith("_Z") and ln.split(";")[0].rstrip().endswith(":")):
        break
    lm = LOC.match(ln)
    if lm:
        cur = (files.get(lm.group(1), lm.group(1)), int(lm.group(2)))
        continue
    bm = BB.match(ln)
    if bm:
        labels[bm.group(1)] = len(body)
        continue
    s = ln.split(";")[0].rstrip()
    if s.strip() and not s.strip().startswith(".") and not s.strip().endswith(":"):
        body.append((s, cur))
loops = []
for k, (s, _) in enumerate(body):
    m = BR.match(s)
    if m:
        t = m.group(1) or m.group(2)
        if t in labels and labels[t] <= k:
            loops.append((labels[t], k))
WB = {"dword": 4, "dwordx2": 8, "dwordx3": 12, "dwordx4": 16, "short": 2, "ushort": 2, "sshort": 2,
      "short_d16_hi": 2, "ubyte": 1, "byte": 1}
agg = collections.OrderedDict()
for k, (s, loc) in enumerate(body):
    m = VM.match(s)
    if not m:
        continue
    w = WB.get(m.group(3), 0)
    inloop = any(b <= k <= e for b, e in loops)
    key = (m.group(2), f"{loc[0]}:{loc[1]}" if loc else "?")
    x = agg.setdefault(key, [0, 0, collections.Counter(), False])
    x[0] += 1
    x[1] += w
    x[2][m.group(3)] += 1
    x[3] = x[3] or inloop
print(f"kernel '{a.kernel}': {len(body)} instructions, {len(loops)} loops")
for (kind, loc), (n, b, ws, inl) in agg.items():
    src = ""
    if a.src and loc != "?":
        fn, line = loc.split(":")
        try:
            src = open(f"{a.src}/{fn}").read().split("\n")[int(line) - 1].strip()[:70]
        except (OSError, IndexError):
            src = ""
    print(f"  {kind:5s} {loc:28s} x{n:<3d} {dict(ws)!s:28s} {'LOOP' if inl else '    '}  {src}")
