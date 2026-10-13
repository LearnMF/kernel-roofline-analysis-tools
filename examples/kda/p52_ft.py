"""kra P-52 for flash-train: vendored kernels synced to the hip_kda production (Gn = exp2f(G_last)),
plus the runner side (forward X1 path and backward produce the decays with the same device exp2f;
the backward keeps a raw Gn copy for the deep-saturation recompute instance).
    python3 p52_ft.py <flash-train tree> <hip_kda tree post-P-52> <hip_kda tree pre-P-52>"""
import sys
from pathlib import Path

FT, NEW, OLD = (Path(a) for a in sys.argv[1:4])
D = FT / "src" / "primitive" / "hygon" / "kda"


def sub(path, old, new, cnt=1):
    t = path.read_text()
    assert t.count(old) == cnt, (path.name, old[:90], t.count(old))
    path.write_text(t.replace(old, new))


# vendored kernels: FT file = 5-line vendor header + the hip_kda file, byte-aligned
for f in ("g2_fwd_v8.cuh", "g2_engine.cuh", "g2_fwd_prep.cuh"):
    ft = (D / "g2" / f).read_text().split("\n")
    head, body = "\n".join(ft[:5]) + "\n", "\n".join(ft[5:])
    assert body == (OLD / "csrc" / "g2" / f).read_text(), f"{f}: flash-train copy is not the pre-P-52 production"
    (D / "g2" / f).write_text(head + (NEW / "csrc" / "g2" / f).read_text())
sub(D / "g2" / "g2_common.h", "__device__ __forceinline__ int pi16(int n) { return 4 * (n & 3) + (n >> 2); }",
    """__device__ __forceinline__ int pi16(int n) { return 4 * (n & 3) + (n >> 2); }
// kra P-52: opaque register copy (ends the source's live range at this point; value unchanged)
__device__ __forceinline__ float g2_vcopy(float x) {
    float y;
    asm volatile("v_mov_b32 %0, %1" : "=v"(y) : "v"(x));
    return y;
}""")

R = D / "g2_runner.cpp"
sub(R, """                 int nseg, const int* ctok, const int* clen, hipStream_t s);""",
    """                 int nseg, const int* ctok, const int* clen, const void* GnRaw, hipStream_t s);
int g2_exp2_inplace(float* x, long n, hipStream_t s);  // kra P-52: Gn -> exp2f(Gn) decays""")
# forward: X1 path to_gn -> decays; the R4 prep writes the decays itself
sub(R, """        check(g2_kda_to_gn(static_cast<const float*>(L.g), L.tab.ctok, L.tab.clen, H, NT, static_cast<float*>(L.Gn),
                           stream),
              "to_gn");""",
    """        check(g2_kda_to_gn(static_cast<const float*>(L.g), L.tab.ctok, L.tab.clen, H, NT, static_cast<float*>(L.Gn),
                           stream),
              "to_gn");
        check(g2_exp2_inplace(static_cast<float*>(L.Gn), static_cast<long>(z.CH * 128), stream), "gn_exp2");""")
sub(R, """                       p.scale, 1, NT * 64, H, L.tab.seg, nseg, L.tab.ctok, L.tab.clen, stream),""",
    """                       p.scale, 1, NT * 64, H, L.tab.seg, nseg, L.tab.ctok, L.tab.clen, nullptr, stream),""")
# backward: decays + a raw copy (the narrow recompute at >= 192 CTAs keeps the production instance)
sub(R, """    void *g, *Gn;
    void *kgB, *HbN;""", """    void *g, *Gn, *GnRaw;
    void *kgB, *HbN;""")
sub(R, """    L.Gn                    = a.take(z.CH * 128 * 4);
    L.kgB""", """    L.Gn                    = a.take(z.CH * 128 * 4);
    L.GnRaw                 = a.take(z.CH * 128 * 4);   // kra P-52
    L.kgB""")
sub(R, """    check(g2_kda_to_gn(static_cast<const float*>(L.g), ctok, clen, H, NT, static_cast<float*>(L.Gn), stream), "to_gn");""",
    """    check(g2_kda_to_gn(static_cast<const float*>(L.g), ctok, clen, H, NT, static_cast<float*>(L.GnRaw), stream), "to_gn");
    check(g2_kda_to_gn(static_cast<const float*>(L.g), ctok, clen, H, NT, static_cast<float*>(L.Gn), stream), "to_gn");
    check(g2_exp2_inplace(static_cast<float*>(L.Gn), static_cast<long>(z.CH * 128), stream), "gn_exp2");  // kra P-52""")
sub(R, """                           NT * 64, H, L.tab.seg, nseg, nullptr, nullptr, stream),""",
    """                           NT * 64, H, L.tab.seg, nseg, nullptr, nullptr, L.GnRaw, stream),""")
print("P-52 FT APPLIED")
