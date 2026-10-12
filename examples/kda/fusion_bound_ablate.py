"""Fusion upper bounds (round 12, step 1): timing-only ablations of the G2 production tree that
remove exactly the HBM traffic a fusion would remove (outputs are WRONG on purpose).  Removed loads
are replaced by synthetic values derived from the pointer bits (not constant -> no MMAC dead-code
elimination); removed stores are guarded by a never-true runtime condition.

    python3 fusion_bound_ablate.py <tree root> <variant>
  F4_s4    : s4a/s4b read no chunk state h / dh (states would stay on chip in a dhu+s4 fusion)
  F4_emit  : fwd_h_bn and dhu write no bN state tensors (HbN / HbN dh)
  F2_s4b   : s4b writes no fp32 dq / dk / dg (the s4 -> tail round trip, write half)
  F2_tail  : bwd_tail reads no fp32 dq / dk / dg (read half)
  F3_rec   : the recurrences (v8o4 fwd_h_o / fwd_h_bn, dhu) read no W/U/K~/Q~/Aqk layout operands"""
import sys
from pathlib import Path

G2 = Path(sys.argv[1]) / "csrc" / "g2"
var = sys.argv[2]


def sub(f, old, new, count=1):
    p = G2 / f
    t = p.read_text()
    assert t.count(old) == count, (f, var, old[:80], t.count(old))
    p.write_text(t.replace(old, new))


def sub_in(f, head, old, new, count=1):
    p = G2 / f
    t = p.read_text()
    i = t.index(head)
    j = t.index("\n}\n", i)
    seg = t[i:j]
    assert seg.count(old) == count, (f, var, head[:40], old[:60], seg.count(old))
    p.write_text(t[:i] + seg.replace(old, new) + t[j:])


SYN = """
// ABLATION helper: a value that depends on the pointer bits but touches no memory
template <typename T>
__device__ __forceinline__ T g2_syn(const void* x) {
    const unsigned s = (unsigned)((reinterpret_cast<unsigned long long>(x) >> 4) & 0x3fu);
    T v{};
    unsigned char* b = reinterpret_cast<unsigned char*>(&v);
    #pragma unroll
    for (int k = 0; k < (int)sizeof(T); ++k) b[k] = (unsigned char)(s + k);
    return v;
}
"""
anchor = "__device__ __forceinline__ size_t g2_nat_blk(int i, int hh, int H, int NT,"
c = (G2 / "g2_common.h").read_text()
(G2 / "g2_common.h").write_text(c.replace(anchor, SYN + "\n" + anchor))

if var == "F4_s4":
    # s4a: P-4 h staging (3 loads)
    sub("g2_wy.cuh", "            f4 hreg = *reinterpret_cast<const f4*>(hsrc);                       // block 0",
        "            f4 hreg = g2_syn<f4>(hsrc);   // ABLATION")
    sub("g2_wy.cuh", "            hreg = *reinterpret_cast<const f4*>(hsrc + 2048);                   // block 1",
        "            hreg = g2_syn<f4>(hsrc + 2048);   // ABLATION")
    sub("g2_wy.cuh", "                        hreg = *reinterpret_cast<const f4*>(hsrc + (kt + 2) * 2048);",
        "                        hreg = g2_syn<f4>(hsrc + (kt + 2) * 2048);   // ABLATION")
    # s4b (split route): initial tiles + per-kt prefetch, inside s4b_tail
    S4BT = "__device__ __forceinline__ void s4b_tail("
    sub_in("g2_wy.cuh", S4BT, "        stg4[(t0 * 2 + 0) * 256 + threadIdx.x] = hg4[t0 * 256 + threadIdx.x];",
           "        stg4[(t0 * 2 + 0) * 256 + threadIdx.x] = g2_syn<int4>(hg4 + t0 * 256 + threadIdx.x);")
    sub_in("g2_wy.cuh", S4BT, "        stg4[(t0 * 2 + 1) * 256 + threadIdx.x] = dhg4[t0 * 256 + threadIdx.x];",
           "        stg4[(t0 * 2 + 1) * 256 + threadIdx.x] = g2_syn<int4>(dhg4 + t0 * 256 + threadIdx.x);")
    sub_in("g2_wy.cuh", S4BT, "            pfh = hg4[(kt + 2) * 256 + threadIdx.x];",
           "            pfh = g2_syn<int4>(hg4 + (kt + 2) * 256 + threadIdx.x);")
    sub_in("g2_wy.cuh", S4BT, "            pfd = dhg4[(kt + 2) * 256 + threadIdx.x];",
           "            pfd = g2_syn<int4>(dhg4 + (kt + 2) * 256 + threadIdx.x);")
elif var == "F4_emit":
    sub("g2_fwd_v8.cuh", """                const size_t off = ((((size_t)i2 * p.H + bh) * 8 + kb) * 8 + (2 * vs + vt)) * 256 + l * 4;
                *reinterpret_cast<v4bh*>(p.HbN + off) = val;""",
        """                const size_t off = ((((size_t)i2 * p.H + bh) * 8 + kb) * 8 + (2 * vs + vt)) * 256 + l * 4;
                if (p.H < 0) *reinterpret_cast<v4bh*>(p.HbN + off) = val;   // ABLATION""")
    sub("g2_engine.cuh", "                *const_cast<v4bh*>(reinterpret_cast<const v4bh*>(p.HbN + off)) = val;",
        "                if (p.H < 0) *const_cast<v4bh*>(reinterpret_cast<const v4bh*>(p.HbN + off)) = val;   // ABLATION",
        count=4)   # every dhu instance (v8b / v8b4 / LIN variants)
elif var == "F2_s4b":
    S4BT = "__device__ __forceinline__ void s4b_tail("
    for nm in ("dq_", "dk_", "dg_"):
        o = {"dq_": "oq", "dk_": "ok", "dg_": "og"}[nm]
        sub_in("g2_wy.cuh", S4BT, f"            *reinterpret_cast<f4*>({nm} + ro + kc) = {o};",
               f"            if (H < 0) *reinterpret_cast<f4*>({nm} + ro + kc) = {o};   // ABLATION")
elif var == "F2_tail":
    sub("g2_bwd_tail.cuh", "                            dyv[w2][u] = *reinterpret_cast<const float2*>(dy + so + u * (p.H * kK) + c2);",
        "                            dyv[w2][u] = g2_syn<float2>(dy + so + u * (p.H * kK) + c2);   // ABLATION")
    sub("g2_bwd_tail.cuh", "? p.dg[((size_t)(t0 + t) * p.H + hh) * kK + d] : 0.f) - runC;",
        "? g2_syn<float>(p.dg + ((size_t)(t0 + t) * p.H + hh) * kK + d) : 0.f) - runC;   // ABLATION", count=2)
elif var == "F3_rec":
    t = (G2 / "g2_fwd_v8.cuh").read_text()
    i = t.index("    auto loadW = [&](int i) {", t.index("const bool hasBN = MODE == 0"))
    j = t.index("    v4bh hx[2];", i)
    seg = t[i:j]
    n = seg.count("ld16x2(")
    seg = seg.replace("ld16x2(", "g2_syn16x2(")
    (G2 / "g2_fwd_v8.cuh").write_text(t[:i] + seg + t[j:])
    t = (G2 / "g2_engine.cuh").read_text()
    i = t.index("    auto loadKA = [&](int i) {", t.index("g2_dhu_v8b_kernel(BwdNParams p)"))
    j = t.index("    auto loadDO = [&](int i) {", i)
    seg = t[i:j]
    m = seg.count("ld16x2(")
    seg = seg.replace("ld16x2(", "g2_syn16x2(")
    (G2 / "g2_engine.cuh").write_text(t[:i] + seg + t[j:])
    c = (G2 / "g2_common.h").read_text()
    helper = """
template <typename P, typename V>
__device__ __forceinline__ void g2_syn16x2(const P* x, V& a, V& b) {   // ABLATION: ld16x2 without memory
    a = g2_syn<V>(x); b = g2_syn<V>(x + 4);
}
"""
    (G2 / "g2_common.h").write_text(c.replace(anchor, helper + "\n" + anchor))
    print("F3_rec replaced", n, "v8o4 and", m, "dhu paired loads")
else:
    raise SystemExit("unknown variant " + var)
print("ABLATE", var)
