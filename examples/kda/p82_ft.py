"""kra P-82 / P-82b flash-train port (bitwise; no ABI change).

Kernels: the vendored g2 sources (byte-aligned with hip_kda) get P-82's kernel parts verbatim -- forward MODE 5
(Ob + HbN) and wu VNEW (v_new = bf_rne(u - W h) from a saved bN(h)) -- executed from p82_save_h.py's own text.
Runner: flash-train's forward / backward are separate stateless calls, so the h hand-over rides the existing OPAQUE
`state` port (bf16 [ftrainKdaStateElements] = 32768 per chunk*head; h needs 16384).  Where the P-82b policy holds
(H <= KDA_G2_SAVE_H_MAX_H = 24, or KDA_G2_V8O4W_MIN_H = 40 <= H <= KDA_G2_SAVE_H_WIDE_MAX_H = 48 with
NT >= KDA_G2_V8O4W_MIN_NT = 512; KDA_G2_SAVE_H=0 disables) a forward given `state` writes only bN(h) there (MODE 5,
instead of MODE 2's bN + Vn + aN) and the backward forms Vn in wu (workspace: + Vn, 256 B/token*head) and skips the
recompute -- the state contract ("results are bit-identical either way") is kept.  Calls without `state` unchanged.
    python3 p82_ft.py <flash-train repo> <p82_save_h.py>"""
import sys
from pathlib import Path

FT = Path(sys.argv[1])
src = Path(sys.argv[2]).read_text()
a0 = src.index("# ---------------------------------------------------------------- A. forward MODE 5")
a1 = src.index('bd = G / "g2_bindings.cpp"')
G = FT / "src" / "primitive" / "hygon" / "kda" / "g2"


def sub(path, old, new, cnt=1):
    s = path.read_text()
    assert s.count(old) == cnt, (path.name, old[:90], s.count(old))
    path.write_text(s.replace(old, new))


exec(compile(src[a0:a1], "p82_save_h.py[A,B]", "exec"), {"G": G, "sub": sub, "Path": Path})

rn = FT / "src" / "primitive" / "hygon" / "kda" / "g2_runner.cpp"
sub(rn, """int g2_wu(const void* k, const void* v, const void* q, const void* beta, const void* A, const void* g, void* Wn,
          void* Wb, void* Un, void* KgA, void* Kn, void* Qn, void* kgB, void* vB, int B, int T, int H, const int* ctok,
          const int* clen, hipStream_t s);""",
    """int g2_wu(const void* k, const void* v, const void* q, const void* beta, const void* A, const void* g, void* Wn,
          void* Wb, void* Un, void* KgA, void* Kn, void* Qn, void* kgB, void* vB, int B, int T, int H, const int* ctok,
          const int* clen, const void* hbn, void* vn_out, hipStream_t s);   // kra P-82: hbn -> v_new (vn_out)""")
sub(rn, """inline bool g2KpLayout(long H) {
    const char* e = getenv("KDA_G2_KPERM_MAX_H");
    return H <= (e ? atol(e) : 20);
}
""", """inline bool g2KpLayout(long H) {
    const char* e = getenv("KDA_G2_KPERM_MAX_H");
    return H <= (e ? atol(e) : 20);
}

inline long g2EnvLong(const char* name, long dflt) {
    const char* e = getenv(name);
    return e ? atol(e) : dflt;
}

// kra P-82/P-82b: shapes where a forward given `state` hands over bN(h) only (MODE 5) and the backward forms v_new
// in wu instead of recomputing the recurrence (hip_kda's saved-h route; bit-identical either way).
inline bool g2SaveH(long H, long NT) {
    if (g2EnvLong("KDA_G2_SAVE_H", 1) == 0) { return false; }
    return H <= g2EnvLong("KDA_G2_SAVE_H_MAX_H", 24) ||
           (g2EnvLong("KDA_G2_V8O4W_MIN_H", 40) <= H && H <= g2EnvLong("KDA_G2_SAVE_H_WIDE_MAX_H", 48) &&
            NT >= g2EnvLong("KDA_G2_V8O4W_MIN_NT", 512));
}
""")
sub(rn, """    L.Vn                    = shape.has_state ? nullptr : a.take(z.CH * 8192 * 2);""",
    """    L.Vn                    = (shape.has_state && !g2SaveH(static_cast<long>(z.H), static_cast<long>(z.NT)))
                                  ? nullptr : a.take(z.CH * 8192 * 2);   // kra P-82: saved-h route forms Vn in wu""")
sub(rn, """    check(g2_fwd_h_v8o(L.Wn, L.Kn, L.Un, L.Gn, L.QgA, L.AqkA, nullptr, p.o, p.state_vn, p.state_hbn, p.state_anv,
                       p.scale, 1, NT * 64, H, L.tab.seg, nseg, L.tab.ctok, L.tab.clen, nullptr, stream),
          "fwd_h_o");""", """    // kra P-82: on the saved-h shapes the state carries bN(h) only (MODE 5); the backward forms v_new in wu.
    const bool hs = p.state_hbn != nullptr && g2SaveH(H, NT);
    check(g2_fwd_h_v8o(L.Wn, L.Kn, L.Un, L.Gn, L.QgA, L.AqkA, nullptr, p.o, hs ? nullptr : p.state_vn, p.state_hbn,
                       hs ? nullptr : p.state_anv, p.scale, 1, NT * 64, H, L.tab.seg, nseg, L.tab.ctok, L.tab.clen,
                       nullptr, stream),
          "fwd_h_o");""")
sub(rn, """    const bool kept = p.state_hbn != nullptr;  // == shape.has_state (the workspace was laid out for it)
    const void* Vn  = kept ? p.state_vn : L.Vn;
    const void* HbN = kept ? p.state_hbn : L.HbN;
    const void* ANv = kept ? p.state_anv : nullptr;  // kra P-22a: s4b rebuilds aN(v_new) from Vn
    check(g2_wu(kn, p.v, qn, p.beta, p.akk, L.g, kept ? nullptr : L.Wn, L.Wb, kept ? nullptr : L.Un, L.KgA,
                kept ? nullptr : L.Kn, L.Qn, L.kgB, nullptr /* kra P-17: no vB */, 1, NT * 64, H, ctok, clen,
                stream),
          "wu");""", """    // kra P-82: saved-h route -- the state holds bN(h) only; wu forms Vn, no recompute.
    const bool hs   = p.state_hbn != nullptr && g2SaveH(H, NT);
    const bool kept = p.state_hbn != nullptr && !hs;  // full B2 state (the workspace was laid out for it)
    const void* Vn  = kept ? p.state_vn : L.Vn;
    const void* HbN = (kept || hs) ? p.state_hbn : L.HbN;
    const void* ANv = kept ? p.state_anv : nullptr;  // kra P-22a: s4b rebuilds aN(v_new) from Vn
    const bool p1   = !kept && !hs;                   // Wn/Un/Kn feed only the recompute
    check(g2_wu(kn, p.v, qn, p.beta, p.akk, L.g, p1 ? L.Wn : nullptr, L.Wb, p1 ? L.Un : nullptr, L.KgA,
                p1 ? L.Kn : nullptr, L.Qn, L.kgB, nullptr /* kra P-17: no vB */, 1, NT * 64, H, ctok, clen,
                hs ? p.state_hbn : nullptr, hs ? L.Vn : nullptr, stream),
          "wu");""")
sub(rn, """    if (!kept) {
        check(g2_fwd_h_v8o(L.Wn, L.Kn, L.Un,""", """    if (p1) {
        check(g2_fwd_h_v8o(L.Wn, L.Kn, L.Un,""")
print("P-82 FT APPLIED")
