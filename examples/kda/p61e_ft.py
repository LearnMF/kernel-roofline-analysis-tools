"""kra P-61e for flash-train: vendored kernels synced to the hip_kda production (prep_a / wu kp layouts, KP recurrence
instances, g2_gn_kp / g2_kp_relayout), plus the runner side:
  * forward X1 chain (no in-kernel l2norm): after to_gn + exp2, Kn / Gn are re-laid out in place when kp;
  * backward: when kp, ONE g2_gn_kp pass replaces the second to_gn + exp2 -- permuted Gn (dhu), natural-k GnNat
    (recompute, new workspace slot of CH*128 fp32, only when kp), GnRaw natural in place (deep-saturation instance).
kp = H <= KDA_G2_KPERM_MAX_H (default 20), the same rule the kernels' launchers apply.
    python3 p61e_ft.py <flash-train tree> <hip_kda tree post-P-61e> <hip_kda tree pre-P-61e>"""
import sys
from pathlib import Path

FT, NEW, OLD = (Path(a) for a in sys.argv[1:4])
D = FT / "src" / "primitive" / "hygon" / "kda"


def sub(path, old, new, cnt=1):
    t = path.read_text()
    assert t.count(old) == cnt, (path.name, old[:90], t.count(old))
    path.write_text(t.replace(old, new))


for f in ("g2_fwd_prep.cuh", "g2_wu.cuh", "g2_fwd_v8.cuh"):
    ft = (D / "g2" / f).read_text().split("\n")
    head, body = "\n".join(ft[:5]) + "\n", "\n".join(ft[5:])
    assert body == (OLD / "csrc" / "g2" / f).read_text(), f"{f}: flash-train copy is not the pre-P-61e production"
    (D / "g2" / f).write_text(head + (NEW / "csrc" / "g2" / f).read_text())

R = D / "g2_runner.cpp"
sub(R, "#include <cstdint>\n#include <cstring>\n", "#include <cstdint>\n#include <cstdlib>\n#include <cstring>\n")
sub(R, "int g2_exp2_inplace(float* x, long n, hipStream_t s);  // kra P-52: Gn -> exp2f(Gn) decays",
    """int g2_exp2_inplace(float* x, long n, hipStream_t s);  // kra P-52: Gn -> exp2f(Gn) decays
int g2_gn_kp(float* raw, float* gp, float* gnat, long n, hipStream_t s);                  // kra P-61e
int g2_kp_relayout(void* kn, long kn_elems, float* gn, long gn_elems, hipStream_t s);     // kra P-61e""")
sub(R, """struct Tables {""", """// kra P-61e: the kp layouts (Kn lane-permuted, natural-k recompute decays) -- same rule as the kernels' launchers
inline bool g2KpLayout(long H) {
    const char* e = getenv("KDA_G2_KPERM_MAX_H");
    return H <= (e ? atol(e) : 20);
}

struct Tables {""")
sub(R, "    void *g, *Gn, *GnRaw;\n", "    void *g, *Gn, *GnRaw, *GnNat;\n")
sub(R, """    L.GnRaw                 = a.take(z.CH * 128 * 4);   // kra P-52
""", """    L.GnRaw                 = a.take(z.CH * 128 * 4);   // kra P-52
    L.GnNat                 = g2KpLayout(static_cast<long>(z.H)) ? a.take(z.CH * 128 * 4) : nullptr;   // kra P-61e
""")
sub(R, """        check(g2_exp2_inplace(static_cast<float*>(L.Gn), static_cast<long>(z.CH * 128), stream), "gn_exp2");
    }""", """        check(g2_exp2_inplace(static_cast<float*>(L.Gn), static_cast<long>(z.CH * 128), stream), "gn_exp2");
        if (g2KpLayout(H)) {  // kra P-61e: the recurrence's kp instances read lane-permuted Kn / natural-k Gn
            check(g2_kp_relayout(L.Kn, static_cast<long>(z.CH * 8192), static_cast<float*>(L.Gn),
                                 static_cast<long>(z.CH * 128), stream),
                  "kp_relayout");
        }
    }""")
sub(R, """    check(g2_kda_to_gn(static_cast<const float*>(L.g), ctok, clen, H, NT, static_cast<float*>(L.Gn), stream), "to_gn");
    check(g2_exp2_inplace(static_cast<float*>(L.Gn), static_cast<long>(z.CH * 128), stream), "gn_exp2");  // kra P-52
    if (!kept) {
        check(g2_fwd_h_v8o(L.Wn, L.Kn, L.Un, L.Gn, nullptr,""", """    const bool kp = g2KpLayout(H);
    if (kp) {  // kra P-61e: permuted Gn (dhu) + natural-k GnNat (recompute); GnRaw -> natural k in place
        check(g2_gn_kp(static_cast<float*>(L.GnRaw), static_cast<float*>(L.Gn), static_cast<float*>(L.GnNat),
                       static_cast<long>(z.CH * 128), stream),
              "gn_kp");
    } else {
        check(g2_kda_to_gn(static_cast<const float*>(L.g), ctok, clen, H, NT, static_cast<float*>(L.Gn), stream), "to_gn");
        check(g2_exp2_inplace(static_cast<float*>(L.Gn), static_cast<long>(z.CH * 128), stream), "gn_exp2");  // kra P-52
    }
    if (!kept) {
        check(g2_fwd_h_v8o(L.Wn, L.Kn, L.Un, kp ? L.GnNat : L.Gn, nullptr,""")
print("P-61e FT APPLIED")
