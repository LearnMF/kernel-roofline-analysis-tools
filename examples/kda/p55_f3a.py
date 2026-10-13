"""kra P-55 (F3a, structural search step 3): s4b recomputes its dAqk operands instead of loading dav's fp32 dAqk.

s4b wave A (rows 16A..16A+15) needs
  aqA[b], b <= A : dAqk[16A+m][16b+4q+x]          (row form)
  aqB[b-A], b >= A : dAqk[16b+4q+x][16A+m]        (column form)
dav computed dAqk[t][j] = scale * sum_vs mmac(do_frag(t rows), vn_frag(j rows)) (8 K-steps from zero), masked
j > t.  Here: aqA[b] = the same MMAC sequence with A = this wave's ado, B = Vn of row block b (avn layout);
aqB[b] = A = this wave's avn, B = do of row block b (mmac_swap_probe: operand-swapped tiles are bit-identical
to the transpose).  The MMAC output lane (m, q) holds [m][q + 4e]; a 4x4 transpose among the lanes
{m, m+16, m+32, m+48} (3 shuffles + selects) gives the [m][4q + x] layout of the loads.  Then * scale and
the production masks.  Wave A loads Vn of blocks 0..A-1 and do of blocks A+1..3 (24 ld8 per lane).
    python3 p55_f3a.py <tree> [probe|full]
      probe: s4b only (dav still writes dAqk; s4b ignores it)
      full : + dav skips pass 1 on the S4 split route (no dAqk write; KDA_G2_F3A=0 keeps the unused write)"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
MODE = sys.argv[2] if len(sys.argv) > 2 else "probe"
P = ROOT / "csrc" / "g2" / "g2_wy.cuh"
t = P.read_text()


def sub(old, new, cnt=1):
    global t
    assert t.count(old) == cnt, (old[:90], t.count(old))
    t = t.replace(old, new)


# 1) drop the aqA / aqB loads (akB stays)
sub("""    f4 aqA[A + 1], aqB[4 - A], akB[4 - A];
    #pragma unroll
    for (int b = 0; b <= A; ++b) {
        aqA[b] = rok ? ld16f(aq_ + row * RA + 16 * b + 4 * q) : f4{0.f, 0.f, 0.f, 0.f};
        if (b == A) {
            #pragma unroll
            for (int x = 0; x < 4; ++x)
                if (4 * q + x > m) aqA[b][x] = 0.f;
        }
    }
    #pragma unroll
    for (int b = A; b < 4; ++b)
        #pragma unroll
        for (int x = 0; x < 4; ++x) {
            const int j = 16 * b + 4 * q + x;
            const bool keep = b > A || 4 * q + x >= m;
            akB[b - A][x] = keep ? dAg[j * RA + row] : 0.f;   // dAT[row][j] == dA[j][row]
            aqB[b - A][x] = (keep && g2_row_ok<M>(j, cl)) ? aq_[j * RA + row] : 0.f;
        }""",
    """    f4 aqA[A + 1], aqB[4 - A], akB[4 - A];
    (void)aq_;
    #pragma unroll
    for (int b = A; b < 4; ++b)
        #pragma unroll
        for (int x = 0; x < 4; ++x) {
            const int j = 16 * b + 4 * q + x;
            const bool keep = b > A || 4 * q + x >= m;
            akB[b - A][x] = keep ? dAg[j * RA + row] : 0.f;   // dAT[row][j] == dA[j][row]
        }""")

# 2) after the own ado / avn loads: other blocks' fragments, the MMAC tiles, transpose, scale, masks
sub("""            for (int vt = 0; vt < 8; ++vt) avn[vt] = g2_c2a_sel(raw_[vt], l);
        }
    }
    const int4* hg4 = reinterpret_cast<const int4*>(hb);
    const int4* dhg4 = reinterpret_cast<const int4*>(dhb);
    int4* stg4 = reinterpret_cast<int4*>(&sm.stg[0][0][0]);""",
    """            for (int vt = 0; vt < 8; ++vt) avn[vt] = g2_c2a_sel(raw_[vt], l);
        }
    }
    {   // kra P-55 (F3a): the dAqk operands recomputed in-CTA (dav pass 1's MMAC sequence, bit-identical)
        auto ld_avn = [&](int b, v4bh (&o)[8]) {
            if (p.v_new != nullptr) {
                const bf16* vnN = p.v_new + g2_nat_blk(i, hh, H, 0, true, 4 * 8 * 256);
                #pragma unroll
                for (int vt = 0; vt < 8; ++vt) o[vt] = ld8(vnN + ((b * 8 + vt) * 64 + l) * 4);
            } else {
                const bf16* vnc = p.vnc + g2_nat_blk(i, hh, H, p.NT, false, 8192) + b * 8 * 256 + l * 4;
                v4bh raw_[8];
                #pragma unroll
                for (int vt = 0; vt < 8; ++vt) raw_[vt] = ld8(vnc + vt * 256);
                #pragma unroll
                for (int vt = 0; vt < 8; ++vt) o[vt] = g2_c2a_sel(raw_[vt], l);
            }
        };
        auto ld_ado = [&](int b, v4bh (&o)[8]) {
            if (p.do_ != nullptr) {
                const bf16* doN = p.do_ + g2_nat_blk(i, hh, H, 0, true, 4 * 8 * 256);
                #pragma unroll
                for (int vt = 0; vt < 8; ++vt) o[vt] = ld8(doN + ((b * 8 + vt) * 64 + l) * 4);
            } else {
                const bool okb = g2_row_ok<M>(16 * b + m, cl);
                const int rob = (16 * b + m) * R + 4 * q;
                #pragma unroll
                for (int vt = 0; vt < 8; ++vt)
                    o[vt] = okb ? ld8(p.do_row + cb + rob + 16 * vt) : v4bh{0, 0, 0, 0};
            }
        };
        // lane (m, q) holds a row of 4 ([m][q + 4e]); return [m][4q + x]: slot x <- lane (m, x)'s element q
        auto xpose4 = [&](const f4 v) {
            f4 o = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int se = (q - r) & 3, dx = (q + r) & 3;
                const float s = se == 0 ? v[0] : se == 1 ? v[1] : se == 2 ? v[2] : v[3];
                const float tv = r == 0 ? s : __shfl(s, m + 16 * dx, 64);
                o[0] = dx == 0 ? tv : o[0];
                o[1] = dx == 1 ? tv : o[1];
                o[2] = dx == 2 ? tv : o[2];
                o[3] = dx == 3 ? tv : o[3];
            }
            return o;
        };
        #pragma unroll
        for (int b = 0; b <= A; ++b) {                 // aqA[b] = dAqk[16A+m][16b + 4q + x]
            v4bh vb[8];
            if (b == A) {
                #pragma unroll
                for (int vt = 0; vt < 8; ++vt) vb[vt] = avn[vt];
            } else {
                ld_avn(b, vb);
            }
            f4 acc = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int vs = 0; vs < 8; ++vs) acc = mmac(ado[vs], vb[vs], acc);
            f4 r4;
            #pragma unroll
            for (int e = 0; e < 4; ++e) r4[e] = acc[e] * p.scale;
            r4 = xpose4(r4);
            #pragma unroll
            for (int x = 0; x < 4; ++x)
                aqA[b][x] = (rok && !(b == A && 4 * q + x > m)) ? r4[x] : 0.f;
        }
        #pragma unroll
        for (int b = A; b < 4; ++b) {                  // aqB[b-A] = dAqk[16b + 4q + x][16A+m]
            v4bh db_[8];
            if (b == A) {
                #pragma unroll
                for (int vt = 0; vt < 8; ++vt) db_[vt] = ado[vt];
            } else {
                ld_ado(b, db_);
            }
            f4 acc = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int vs = 0; vs < 8; ++vs) acc = mmac(avn[vs], db_[vs], acc);
            f4 r4;
            #pragma unroll
            for (int e = 0; e < 4; ++e) r4[e] = acc[e] * p.scale;
            r4 = xpose4(r4);
            #pragma unroll
            for (int x = 0; x < 4; ++x) {
                const int j = 16 * b + 4 * q + x;
                const bool keep = b > A || 4 * q + x >= m;
                aqB[b - A][x] = (keep && g2_row_ok<M>(j, cl)) ? r4[x] : 0.f;
            }
        }
    }
    const int4* hg4 = reinterpret_cast<const int4*>(hb);
    const int4* dhg4 = reinterpret_cast<const int4*>(dhb);
    int4* stg4 = reinterpret_cast<int4*>(&sm.stg[0][0][0]);""")
P.write_text(t)

if MODE == "full":
    def fsub(path, old, new, cnt=1):
        x = path.read_text()
        assert x.count(old) == cnt, (path.name, old[:80], x.count(old))
        path.write_text(x.replace(old, new))
    B = ROOT / "csrc" / "g2" / "g2_bindings.cpp"
    fsub(B, """                               c10::optional<torch::Tensor> clen, bool emit_ando) {
    TORCH_CHECK(do_.dim() == 4 && do_.size(0) == 1, "g2_dav: dense B=1 only");""",
         """                               c10::optional<torch::Tensor> clen, bool emit_ando, bool emit_daqk) {
    TORCH_CHECK(do_.dim() == 4 && do_.size(0) == 1, "g2_dav: dense B=1 only");""")
    fsub(B, """    auto dA = torch::empty({1, Tr, heads, 64},
                           do_.options().dtype(torch::kFloat32));      // FLA: fp32 dAqk""",
         """    // kra P-55: emit_daqk=false (S4 split route): s4b recomputes dAqk, pass 1 is skipped (dA == nullptr)
    auto dA = emit_daqk ? torch::empty({1, Tr, heads, 64}, do_.options().dtype(torch::kFloat32))
                        : torch::empty({0}, do_.options().dtype(torch::kFloat32));""")
    fsub(B, """    const int rc = g2_dav(do_.data_ptr(), Vn.data_ptr(), A.data_ptr(), dA.data_ptr(),""",
         """    const int rc = g2_dav(do_.data_ptr(), Vn.data_ptr(), A.data_ptr(), emit_daqk ? dA.data_ptr() : nullptr,""")
    fsub(B, """          py::arg("emit_ando") = true);
    m.def("intra_bwd", &intra_bwd,""",
         """          py::arg("emit_ando") = true, py::arg("emit_daqk") = true);
    m.def("intra_bwd", &intra_bwd,""")
    O = ROOT / "hip_kda" / "ops" / "g2_ops.py"
    fsub(O, """def dav(do, Vn, A, scale, heads, chunks, rowmap=None, emit_ando=True):""",
         """def dav(do, Vn, A, scale, heads, chunks, rowmap=None, emit_ando=True, emit_daqk=True):""")
    fsub(O, """                      int(heads), int(chunks), *_rowmap(rowmap)[:2], bool(emit_ando))""",
         """                      int(heads), int(chunks), *_rowmap(rowmap)[:2], bool(emit_ando), bool(emit_daqk))""")
    K = ROOT / "hip_kda" / "ops" / "k3_g2.py"
    fsub(K, """        dAqk, dVn, dOT, aN_do = g2_ops.dav(do, Vn, Aqk, scale, H, NT, rowmap=lay,
                                           emit_ando=not no_ando)""",
         """        # kra P-55 (F3a): on the S4 split route s4b recomputes dAqk, so dav skips pass 1 (KDA_G2_F3A=0 keeps
        # dav's now-unused dAqk write, an A/B hook for the dav side)
        f3a = no_vb and os.environ.get("KDA_G2_F3A", "1").strip() != "0"
        dAqk, dVn, dOT, aN_do = g2_ops.dav(do, Vn, Aqk, scale, H, NT, rowmap=lay,
                                           emit_ando=not no_ando, emit_daqk=not f3a)""")
print("P-55", MODE, "APPLIED")
