"""kra P-93 evaluation tree (NOT bitwise, constraint-relaxation option B1): the s4b -> bwd_tail handoff of dq / dk / dg
in bf16 instead of fp32 (16K/12: 604 MB -> 302 MB of HBM traffic).  Everything else, including every accumulation,
stays fp32; s4b rounds its fp32 results once (RNE) before the store, the tail widens them back.
    python3 p93_b16_handoff.py <tree>
Used only to measure (a) the time saved and (b) the extra error against the fp32 reference
(tests/g2/fp32_ref_check.py: ours vs ref, FLA vs ref)."""
import sys
from pathlib import Path

t = Path(sys.argv[1])
qk_only = "--qk-only" in sys.argv   # B1': dq / dk in bf16, dg stays fp32 (dg feeds the reverse cumsum and the dt_bias sum)


def edit(rel, pairs):
    f = t / rel
    s = f.read_text()
    for old, new, cnt in pairs:
        assert s.count(old) == cnt, (rel, old[:80], s.count(old))
        s = s.replace(old, new)
    f.write_text(s)


edit("csrc/g2/g2_bindings.cpp", [
    ("""    auto dq = torch::empty({B, Tr, H, kK}, fopts);
    auto dk = torch::empty({B, Tr, H, kK}, fopts);
    auto dg = torch::empty({B, Tr, H, kK}, fopts);""",
     """    auto dq = torch::empty({B, Tr, H, kK}, bopts);   // kra P-93: bf16 handoff to the tail
    auto dk = torch::empty({B, Tr, H, kK}, bopts);
    auto dg = torch::empty({B, Tr, H, kK}, """ + ("fopts" if qk_only else "bopts") + """);""", 1),
    ("""    check_f32(dq, "bwd_tail dq"); check_f32(dk, "bwd_tail dk");
    check_f32(dg, "bwd_tail dg"); check_f32(db, "bwd_tail db");""",
     """    check_bf16(dq, "bwd_tail dq"); check_bf16(dk, "bwd_tail dk");   // kra P-93
    """ + ("check_f32" if qk_only else "check_bf16") + """(dg, "bwd_tail dg"); check_f32(db, "bwd_tail db");""", 1)])
edit("csrc/g2/g2_wy.cuh", [
    ("""    float *dq_ = p.dq + cb, *dk_ = p.dk + cb, *dg_ = p.dg + cb;
    const float* aq_ = p.dAqk + cbA;
    const float* dAg = p.dA + cbAp;
    const bf16* dwTg = p.dwT + g2_nat_blk(i, hh, H, 0, true, kD * kBT);""",
     """    short *dq_ = reinterpret_cast<short*>(p.dq) + cb, *dk_ = reinterpret_cast<short*>(p.dk) + cb,
          *dg_b_ = reinterpret_cast<short*>(p.dg) + cb;   // kra P-93: bf16 handoff
    float* dg_f_ = p.dg + cb;
    const float* aq_ = p.dAqk + cbA;
    const float* dAg = p.dA + cbAp;
    const bf16* dwTg = p.dwT + g2_nat_blk(i, hh, H, 0, true, kD * kBT);""", 1),
    ("""        if (rok) {
            *reinterpret_cast<f4*>(dq_ + ro + kc) = oq;
            *reinterpret_cast<f4*>(dk_ + ro + kc) = ok;
            *reinterpret_cast<f4*>(dg_ + ro + kc) = og;
        }""",
     """        if (rok) {   // kra P-93: one RNE rounding of the fp32 results
            *reinterpret_cast<v4bh*>(dq_ + ro + kc) = v4bh{bf_rne(oq[0]), bf_rne(oq[1]), bf_rne(oq[2]), bf_rne(oq[3])};
            *reinterpret_cast<v4bh*>(dk_ + ro + kc) = v4bh{bf_rne(ok[0]), bf_rne(ok[1]), bf_rne(ok[2]), bf_rne(ok[3])};
            """ + ("*reinterpret_cast<f4*>(dg_f_ + ro + kc) = og;" if qk_only else
                   "*reinterpret_cast<v4bh*>(dg_b_ + ro + kc) = v4bh{bf_rne(og[0]), bf_rne(og[1]), bf_rne(og[2]), bf_rne(og[3])};") + """
        }""", 1)])
edit("csrc/g2/g2_bwd_tail.cuh", [
    ("""    const float* dq; const float* dk; const float* dg; const float* db;""",
     """    const short* dq; const short* dk; const """ + ("float" if qk_only else "short") + """* dg; const float* db;   // kra P-93""", 1)] +
    ([] if qk_only else [
    ("""                                     ? p.dg[((size_t)(t0 + t) * p.H + hh) * kK + d] : 0.f) - runC;""",
     """                                     ? bf2f(p.dg[((size_t)(t0 + t) * p.H + hh) * kK + d]) : 0.f) - runC;""", 1),
    ("""                    ? *reinterpret_cast<const float2*>(p.dg + ((size_t)(t0 + t) * p.H + hh) * kK + d0)""",
     """                    ? p93_ld2b(p.dg + ((size_t)(t0 + t) * p.H + hh) * kK + d0)""", 1)]) + [
    ("""                const float* dy = which ? p.dk : p.dq;""", """                const short* dy = which ? p.dk : p.dq;""", 1),
    ("""                    const float2 dyv = *reinterpret_cast<const float2*>(dy + so + c2);""",
     """                    const float2 dyv = p93_ld2b(dy + so + c2);""", 1),
    ("""                    const float* dy = w2 ? p.dk : p.dq;""", """                    const short* dy = w2 ? p.dk : p.dq;""", 1),
    ("""                            dyv[w2][u] = *reinterpret_cast<const float2*>(dy + so + u * (p.H * kK) + c2);""",
     """                            dyv[w2][u] = p93_ld2b(dy + so + u * (p.H * kK) + c2);""", 1),
    ("""    g2::TailParams p{(const float*)dq, (const float*)dk, (const float*)dg,""",
     """    g2::TailParams p{(const short*)dq, (const short*)dk, (const """ + ("float" if qk_only else "short") + """*)dg,""", 1),
    ("""struct TailParams {""", """__device__ __forceinline__ float2 p93_ld2b(const short* q) {   // kra P-93: 2 bf16 -> 2 fp32
    const short2 s_ = *reinterpret_cast<const short2*>(q);
    return float2{bf2f(s_.x), bf2f(s_.y)};
}
struct TailParams {""", 1)])
print("P-93 APPLIED" + (" (dq/dk only)" if qk_only else ""))
