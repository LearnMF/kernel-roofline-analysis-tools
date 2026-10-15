"""kra P-62 for flash-train: vendored g2_fwd_prep.cuh / g2_fwd_v8.cuh synced to the hip_kda production (prep_b Un
in the pm column order when kp, VP forward instances, g2_kp_relayout also re-lays Un out), plus the runner's X1
call passing Un.
    python3 p62_ft.py <flash-train tree> <hip_kda tree post-P-62> <hip_kda tree pre-P-62>"""
import sys
from pathlib import Path

FT, NEW, OLD = (Path(a) for a in sys.argv[1:4])
D = FT / "src" / "primitive" / "hygon" / "kda"


def sub(path, old, new, cnt=1):
    t = path.read_text()
    assert t.count(old) == cnt, (path.name, old[:90], t.count(old))
    path.write_text(t.replace(old, new))


for f in ("g2_fwd_prep.cuh", "g2_fwd_v8.cuh"):
    ft = (D / "g2" / f).read_text().split("\n")
    head, body = "\n".join(ft[:5]) + "\n", "\n".join(ft[5:])
    assert body == (OLD / "csrc" / "g2" / f).read_text(), f"{f}: flash-train copy is not the pre-P-62 production"
    (D / "g2" / f).write_text(head + (NEW / "csrc" / "g2" / f).read_text())
for f in ("g2_wu.cuh",):   # unchanged by P-62: must still be byte-aligned
    ft = (D / "g2" / f).read_text().split("\n")
    assert "\n".join(ft[5:]) == (NEW / "csrc" / "g2" / f).read_text(), f"{f}: not aligned"

R = D / "g2_runner.cpp"
sub(R, "int g2_kp_relayout(void* kn, long kn_elems, float* gn, long gn_elems, hipStream_t s);     // kra P-61e",
    "int g2_kp_relayout(void* kn, long kn_elems, float* gn, long gn_elems, void* un, hipStream_t s);   // kra P-61e, P-62")
sub(R, """            check(g2_kp_relayout(L.Kn, static_cast<long>(z.CH * 8192), static_cast<float*>(L.Gn),
                                 static_cast<long>(z.CH * 128), stream),""",
    """            check(g2_kp_relayout(L.Kn, static_cast<long>(z.CH * 8192), static_cast<float*>(L.Gn),
                                 static_cast<long>(z.CH * 128), L.Un, stream),   // kra P-62: + Un""")
print("P-62 FT APPLIED")
