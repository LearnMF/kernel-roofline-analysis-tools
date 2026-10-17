"""Map sqtt_hotspots wait sites to source lines (rule R58).

Reads the 'wait sites' block of a sqtt_hotspots.py report (site = wait <- prev <- prev2, as asm text) and a device ISA
compiled with -g (isa_g2_dbg.sh with ISA_EXTRA=-g; .loc directives), finds the kernel's function body, and matches
the (prev2, prev, wait) instruction triple -- exactly first, then with register numbers / immediates normalized --
reporting the source file:line of the .loc in effect at the wait and at the two preceding instructions.
    python3 map_waits.py <report.txt> <isa_g.s> <kernel-symbol-fragment> [--top 12]"""
import argparse
import re

ap = argparse.ArgumentParser()
ap.add_argument("report")
ap.add_argument("isa")
ap.add_argument("kernel")
ap.add_argument("--top", type=int, default=12)
a = ap.parse_args()

SITE = re.compile(r"^\s+([\d.]+)%\s+(\d+)\s+x\s*(\d+)\s+(.*)$")
sites = []
in_block = False
for ln in open(a.report):
    if ln.startswith("wait sites"):
        in_block = True
        continue
    if in_block:
        m = SITE.match(ln)
        if not m:
            if ln.strip() == "" or ln.startswith("wait charged") or ln.startswith("issue delay"):
                if sites:
                    break
            continue
        parts = [p.strip() for p in m.group(4).split(" <- ")]
        sites.append((float(m.group(1)), int(m.group(3)), parts))
sites = sites[:a.top]

files, body, cur_loc, started = {}, [], None, False
FILE = re.compile(r'^\s*\.file\s+(\d+)\s+"([^"]*)"(?:\s+"([^"]*)")?')
LOC = re.compile(r"^\s*\.loc\s+(\d+)\s+(\d+)")
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
        cur_loc = (files.get(lm.group(1), lm.group(1)), int(lm.group(2)))
        continue
    s = ln.split(";")[0].strip()
    if not s or s.startswith(".") or s.endswith(":"):
        continue
    body.append((s, cur_loc))


def norm(x):
    x = re.sub(r"\b([vs])\[\d+:\d+\]", r"\1[R]", x)
    x = re.sub(r"\b([vs])\d+\b", r"\1R", x)
    x = re.sub(r"0x[0-9a-fA-F]+|\b\d+\b", "N", x)
    return re.sub(r"\s+", " ", x)


def find(trip, f):
    w, p1, p2 = (f(t) for t in trip)
    hits = []
    for k in range(2, len(body)):
        if f(body[k][0]) == w and f(body[k - 1][0]) == p1 and f(body[k - 2][0]) == p2:
            hits.append(k)
    return hits


print(f"kernel '{a.kernel}': {len(body)} instructions in the -g ISA")
for pct, cnt, parts in sites:
    trip = (parts + ["", "", ""])[:3]
    hits = find(trip, lambda x: re.sub(r"\s+", " ", x)) or find(trip, norm)
    where = sorted({f"{body[k][1][0]}:{body[k][1][1]}" if body[k][1] else "?" for k in hits})
    prev = sorted({f"{body[k - 1][1][0]}:{body[k - 1][1][1]}" if body[k - 1][1] else "?" for k in hits})
    print(f"{pct:5.1f}% x{cnt:<6d} {trip[0][:34]:34s} -> {', '.join(where[:4]) or 'NO MATCH'}"
          f"   (prev at {', '.join(prev[:3])})  [{len(hits)} match]")
