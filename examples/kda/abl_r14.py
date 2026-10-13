"""Round-14 batch: one real change (hreuse, bit-identical) + timing-only ablations (outputs WRONG on
purpose) that bound what each remaining candidate could save (rule R42: ablate before building).

    python3 abl_r14.py <tree> <variant>
variants:
  hreuse      [real, kra P-53] forward recurrence (narrow v8o4): the GEMM1 h fragments are kept in
              registers for the o pass (it reloaded the same 16 fragments from LDS: 8 ds_read2_b64
              per wave per step); same operands, same order -> bit-identical
  fwdh_noconf narrow v8o4: the h / v_new transposed ds_write_b16 go to one dword per lane (no bank
              conflicts, same instruction count)
  fwdh_noconf3 fwdh_noconf with distinct slots per (kb, vt, e): fwdh_noconf/2 let the compiler drop the
              kb0 write_h as a dead store and with it the ht0 GEMM2 chain (8 MMAC/step) -> INVALID bounds
  fwdh_noconf2 as fwdh_noconf but lane pairs share a dword (bank-conflict-free, sub-dword writes kept)
  rnd_trunc   narrow v8o4 + dhu (8-wave, 4-wave): bf16 RNE rounding (v_bfe + v_add3) -> truncation
  bn_wide     narrow v8o4 (MODE 4) + dhu: the bN emit's 4 scalar ds_read_u16 per quad -> one b64 read
  dhu_st16    dhu 4-wave: the two 8 B bN stores per K-block -> one 16 B store (contiguous, wrong layout)
  wu_noconf   g2_wu: every LDS scalar store (phase-1/2 fills, B-form tile, phase-3 tiles) to lane-consecutive
              per-(thread, element) slots -- bank-conflict-free, same instruction count, no dead stores
  tail_nosum  bwd_tail: tid 0's sequential 512-term dA partial sum reduced to 2 terms
  prep_noR    prep_a's fp32 R scratch write + prep_b's R reads removed"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
var = sys.argv[2]
G2 = ROOT / "csrc" / "g2"


def region_sub(path, start, end, old, new, cnt):
    t = path.read_text()
    a = t.index(start)
    b = t.index(end, a)
    body = t[a:b]
    assert body.count(old) == cnt, (path.name, var, old[:70], body.count(old))
    path.write_text(t[:a] + body.replace(old, new) + t[b:])


F8 = G2 / "g2_fwd_v8.cuh"
EN = G2 / "g2_engine.cuh"
V8O4 = ("g2_fwd_h_v8o4_kernel(FwdOParams p) {", "g2_fwd_h_v8o4w_kernel(FwdOParams p) {")
DHU = ("g2_dhu_v8b_kernel(BwdNParams p) {", 'extern "C" int g2_dhu_w8b(')
DHU4 = ("g2_dhu_v8b4_kernel(BwdNParams p) {", 'extern "C" int g2_dhu_w8b(')

if var == "hreuse":
    region_sub(F8, *V8O4,
               """        f4 acc0 = f4{0.f,0.f,0.f,0.f}, acc1 = f4{0.f,0.f,0.f,0.f};
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
            acc0 = mmac(wf[kt], ld8(hB + (0 * 16 + m) * kPadK + 16 * kt + q4 * 4), acc0);
            acc1 = mmac(wf[kt], ld8(hB + (1 * 16 + m) * kPadK + 16 * kt + q4 * 4), acc1);
        }""",
               """        f4 acc0 = f4{0.f,0.f,0.f,0.f}, acc1 = f4{0.f,0.f,0.f,0.f};
        v4bh hbf0[8], hbf1[8];   // kra P-53: h fragments kept for the o pass (hB is unchanged until write_h)
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
            hbf0[kt] = ld8(hB + (0 * 16 + m) * kPadK + 16 * kt + q4 * 4);
            hbf1[kt] = ld8(hB + (1 * 16 + m) * kPadK + 16 * kt + q4 * 4);
            acc0 = mmac(wf[kt], hbf0[kt], acc0);
            acc1 = mmac(wf[kt], hbf1[kt], acc1);
        }""", 1)
    region_sub(F8, *V8O4,
               """                oh0 = mmac(qgf[kt], ld8(hB + (0 * 16 + m) * kPadK + 16 * kt + q4 * 4), oh0);
                oh1 = mmac(qgf[kt], ld8(hB + (1 * 16 + m) * kPadK + 16 * kt + q4 * 4), oh1);""",
               """                oh0 = mmac(qgf[kt], hbf0[kt], oh0);   // kra P-53: GEMM1's fragments
                oh1 = mmac(qgf[kt], hbf1[kt], oh1);""", 1)
elif var == "fwdh_noconf":
    region_sub(F8, *V8O4,
               "reinterpret_cast<short*>(hB)[(16 * vt + m) * kPadK + 16 * kb + c_col(l, e)] = hx[vt][e];",
               "reinterpret_cast<short*>(hB)[((vt * 4 + e) & 7) * 512 + wv * 128 + 2 * l] = hx[vt][e];   // ABLATION", 1)
    region_sub(F8, *V8O4,
               "reinterpret_cast<short*>(vB)[(16 * 0 + c_col(l, e)) * kPadT + 16 * r1 + m] = x0[e];",
               "reinterpret_cast<short*>(vB)[((0 * 4 + e) & 7) * 256 + 2 * l] = x0[e];   // ABLATION", 1)
    region_sub(F8, *V8O4,
               "reinterpret_cast<short*>(vB)[(16 * 1 + c_col(l, e)) * kPadT + 16 * r1 + m] = x1[e];",
               "reinterpret_cast<short*>(vB)[((1 * 4 + e) & 7) * 256 + 2 * l] = x1[e];   // ABLATION", 1)
elif var == "fwdh_noconf3":   # as fwdh_noconf, but every (kb, vt, e) keeps its own slot (no dead stores)
    region_sub(F8, *V8O4,
               "reinterpret_cast<short*>(hB)[(16 * vt + m) * kPadK + 16 * kb + c_col(l, e)] = hx[vt][e];",
               "reinterpret_cast<short*>(hB)[((((kb >> 2) * 2 + vt) * 4 + e) * 256) + 2 * l] = hx[vt][e];   // ABLATION", 1)
    region_sub(F8, *V8O4,
               "reinterpret_cast<short*>(vB)[(16 * 0 + c_col(l, e)) * kPadT + 16 * r1 + m] = x0[e];",
               "reinterpret_cast<short*>(vB)[((0 * 4 + e) & 7) * 256 + 2 * l] = x0[e];   // ABLATION", 1)
    region_sub(F8, *V8O4,
               "reinterpret_cast<short*>(vB)[(16 * 1 + c_col(l, e)) * kPadT + 16 * r1 + m] = x1[e];",
               "reinterpret_cast<short*>(vB)[((1 * 4 + e) & 7) * 256 + 2 * l] = x1[e];   // ABLATION", 1)
elif var == "fwdh_noconf2":   # bank-conflict-free but lane pairs still share a dword (sub-dword writes kept)
    region_sub(F8, *V8O4,
               "reinterpret_cast<short*>(hB)[(16 * vt + m) * kPadK + 16 * kb + c_col(l, e)] = hx[vt][e];",
               "reinterpret_cast<short*>(hB)[((vt * 4 + e) & 7) * 512 + wv * 128 + l] = hx[vt][e];   // ABLATION", 1)
    region_sub(F8, *V8O4,
               "reinterpret_cast<short*>(vB)[(16 * 0 + c_col(l, e)) * kPadT + 16 * r1 + m] = x0[e];",
               "reinterpret_cast<short*>(vB)[((0 * 4 + e) & 7) * 256 + l] = x0[e];   // ABLATION", 1)
    region_sub(F8, *V8O4,
               "reinterpret_cast<short*>(vB)[(16 * 1 + c_col(l, e)) * kPadT + 16 * r1 + m] = x1[e];",
               "reinterpret_cast<short*>(vB)[((1 * 4 + e) & 7) * 256 + l] = x1[e];   // ABLATION", 1)
elif var == "rnd_trunc":
    t = (G2 / "g2_common.h").read_text()
    anchor = "__device__ __forceinline__ short bfr(float x) { return bf_rne(x); }"
    assert t.count(anchor) == 1
    (G2 / "g2_common.h").write_text(t.replace(anchor, anchor +
        "\n__device__ __forceinline__ short bf_trunc(float x) { return (short)(__float_as_uint(x) >> 16); }   // ABLATION"))
    for path, reg in ((F8, V8O4), (EN, DHU)):
        t = path.read_text()
        a = t.index(reg[0]); b = t.index(reg[1], a)
        n = t[a:b].count("bf_rne(")
        assert n >= 3, (path.name, n)
        path.write_text(t[:a] + t[a:b].replace("bf_rne(", "bf_trunc(") + t[b:])
elif var == "bn_wide":
    region_sub(F8, *V8O4,
               "val[e] = reinterpret_cast<const short*>(hB)[(16 * vt + 4 * q4 + e) * kPadK + 16 * kb + pi16(m)];",
               "val[e] = ld8(hB + (16 * vt + m) * kPadK + 16 * kb + q4 * 4)[e];   // ABLATION", 1)
    region_sub(EN, *DHU,
               "val[e] = reinterpret_cast<const short*>(dhB)[(16 * vt + 4 * q4_ + e) * kPadK + 16 * wv + pi16(m_)];",
               "val[e] = ld8(dhB + (16 * vt + m_) * kPadK + 16 * wv + q4_ * 4)[e];   // ABLATION", 2)
    region_sub(EN, *DHU4,
               "val[e] = reinterpret_cast<const short*>(dhB)[(16 * vt + 4 * q4_ + e) * kPadK + 16 * kb + pi16(m_)];",
               "val[e] = ld8(dhB + (16 * vt + m_) * kPadK + 16 * kb + q4_ * 4)[e];   // ABLATION", 1)
elif var == "dhu_st16":
    region_sub(EN, *DHU4, "    auto bN_emit = [&](int i, int kb) {\n",
               "    auto bN_emit = [&](int i, int kb) {\n        v4bh pend{};   // ABLATION\n", 1)
    region_sub(EN, *DHU4,
               "                *const_cast<v4bh*>(reinterpret_cast<const v4bh*>(p.HbN + off)) = val;",
               "                if (vt == 0) pend = val;   // ABLATION: one 16 B store per K-block\n"
               "                else st16x2(const_cast<bf16*>(p.HbN) + (off - 256 - l * 4) + l * 8, pend, val);", 1)
elif var == "wu_noconf":   # every LDS scalar store of g2_wu to lane-consecutive, per-(thread, element) distinct slots
    WU = G2 / "g2_wu.cuh"
    t = WU.read_text()
    reps = [
        ("""                slab[(size_t)sr * kPadJ2 + wu_slab_col(sr, j)] =
                    bf_rne(bf2f(kv[dv]) * b * eg[dv]);""",
         """                slab[(it * 4 + dv) * 256 + tid] =   // ABLATION
                    bf_rne(bf2f(kv[dv]) * b * eg[dv]);""", 1),
        ("""                slab[(size_t)(vq + dv) * kPadJ2 + wu_slab_col(vq + dv, j)] = bf_rne(bf2f(vv[dv]) * b);""",
         """                slab[(it_ * 4 + dv) * 256 + tid] = bf_rne(bf2f(vv[dv]) * b);   // ABLATION""", 2),
        ("""                    for (int e = 0; e < 4; ++e) tile[wu_tile_idx(ib + 4 * e)] = we8[vt][e];""",
         """                    for (int e = 0; e < 4; ++e) tile[(vt * 4 + e) * 256 + tid + 0 * ib] = we8[vt][e];   // ABLATION""", 1),
        ("""                        tile[wu_tile_idx(ibn_of(tt) + 4 * dv)] = kgr[it][dv];""",
         """                        tile[(it * 4 + dv) * 256 + tid] = kgr[it][dv];   // ABLATION""", 1),
        ("""                    for (int dv = 0; dv < 4; ++dv) tile[wu_tile_idx(ibn_of(tid + 256 * it) + 4 * dv)] = qst[it][dv];""",
         """                    for (int dv = 0; dv < 4; ++dv) tile[(it * 4 + dv) * 256 + tid] = qst[it][dv];   // ABLATION""", 1),
    ]
    for o, n, c in reps:
        assert t.count(o) == c, (o[:60], t.count(o))
        t = t.replace(o, n)
    WU.write_text(t)
elif var == "tail_nosum":   # bwd_tail: tid 0's sequential 512-term dA_part sum -> 2 terms (timing only)
    TL = G2 / "g2_bwd_tail.cuh"
    t = TL.read_text()
    o = """            for (int k = 0; k < 8 * 64; ++k) s += sDA[k];"""
    assert t.count(o) == 1
    TL.write_text(t.replace(o, """            for (int k = 0; k < 2; ++k) s += sDA[k * (8 * 64 - 1)];   // ABLATION"""))
elif var == "prep_noR":
    PR = G2 / "g2_fwd_prep.cuh"
    t = PR.read_text()
    old_w = "        for (int j = threadIdx.x; j < 1280; j += 256) d4[j] = s4[j];"
    old_r = "        const float* r = Rg + b * 256 + 4 * m + q4;\n        return f4{r[0], r[64], r[128], r[192]};"
    assert t.count(old_w) == 1 and t.count(old_r) == 1
    t = t.replace(old_w, "        for (int j = threadIdx.x; j < 1280; j += 256) if (p.scale == 12345.f) d4[j] = s4[j];   // ABLATION")
    t = t.replace(old_r, "        (void)Rg; return f4{(float)b, (float)m, (float)q4, 1.f};   // ABLATION")
    PR.write_text(t)
else:
    raise SystemExit(f"unknown variant {var}")
print("R14", var)
