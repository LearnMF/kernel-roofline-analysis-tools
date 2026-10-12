"""kra P-47 = B4b-lite (on top of P-45c): the fused tail with TA-efficient access patterns.

The 8-wave register-resident form of the B4b design (dq/dk kept in VGPRs) fails its own budget gate on
gfx936 (4-wave: +64 VGPRs -> ~300; 8-wave column split duplicates the s4b core; 8-wave V split
duplicates the per-kt q/k/g loads, ~+1/3 TA).  B4b-lite keeps the production 4-wave / 2-CTA s4b and
fixes what made P-45 slow (R36, ta_microbench):
  * epilogue: each wave re-reads ITS OWN 16 rows of fp32 dq/dk with the bwd_tail kernel's row mapping
    (lane = 2 columns, 512 B per row instruction instead of 16 rows x 64 B, ~8x cheaper per byte) and
    the tail's exact operation order (pair products, xor-tree row dot) -> dq/dk bit-identical to the
    unfused path
  * dt_bias (128 floats per head) lives in the TQ/TK table pad rows (64..67, never touched by the
    tables), loaded once in the prologue -- 4 VGPRs and one VMEM instruction per kt less
  * the ddt column partials: one store instruction per kt (lane m == e stores column 4q+e) instead of 4
dg stays as P-45 (in-loop reverse suffix + gate backward, B-class association).
python3 p47_s4bt_lite.py <tree with P-45c>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wy.cuh"
t = P.read_text()


def rep(old, new):
    global t
    assert t.count(old) == 1, (old[:90], t.count(old))
    t = t.replace(old, new)


# ---- dt_bias -> table pad rows (TQ for columns 0..63, TK for 64..127), loaded once before the loop
rep("""    float dAs = 0.f, Ag = 0.f;                                 // kra P-45 (TAIL)
    const short* gr_ = nullptr;
    if constexpr (TAIL) {
        Ag = __int_as_float(__builtin_amdgcn_readfirstlane(__float_as_int(expf(p.A_log[hh]))));
        gr_ = p.g_raw + cb;
    }""", """    float dAs = 0.f, Ag = 0.f;                                 // kra P-45 (TAIL)
    const short* gr_ = nullptr;
    // kra P-47: dt_bias column c at pad slot ((c & 15), 64 + ((c >> 4) & 3)) of TQ (c < 64) / TK (c >= 64)
    auto dtb_slot = [&](int c) -> float* {
        return (c < 64 ? sm.TQ : sm.TK) + (c & 15) * TS + 64 + ((c >> 4) & 3);
    };
    if constexpr (TAIL) {
        Ag = __int_as_float(__builtin_amdgcn_readfirstlane(__float_as_int(expf(p.A_log[hh]))));
        gr_ = p.g_raw + cb;
        if (threadIdx.x < kD) *dtb_slot(threadIdx.x) = p.dt_bias[hh * kD + threadIdx.x];
    }""")
rep("""            grw = *reinterpret_cast<const v4bh*>(gr_ + rog + kc);   // kra P-45b: in flight across the barrier
            dtb = ld16f(p.dt_bias + hh * kD + kc + 4 * q);
""", """            grw = *reinterpret_cast<const v4bh*>(gr_ + rog + kc);   // kra P-45b: in flight across the barrier
            #pragma unroll
            for (int e = 0; e < 4; ++e) dtb[e] = *dtb_slot(kc + 4 * q + e); // kra P-47: from LDS
""")
# ---- merged ddt partial store
rep("""            #pragma unroll
            for (int e = 0; e < 4; ++e) {                    // ddt column partial over the wave's rows
                float v = cv[e];
                v += __shfl_xor(v, 1, 16); v += __shfl_xor(v, 2, 16);
                v += __shfl_xor(v, 4, 16); v += __shfl_xor(v, 8, 16);
                if (m == 0) p.ddt_part4[(((size_t)hh * kD + kc + 4 * q + e) * p.NT + i) * 4 + A] = v;
            }""", """            float vsel = 0.f;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {                    // ddt column partial over the wave's rows
                float v = cv[e];
                v += __shfl_xor(v, 1, 16); v += __shfl_xor(v, 2, 16);
                v += __shfl_xor(v, 4, 16); v += __shfl_xor(v, 8, 16);
                vsel = (m == e) ? v : vsel;
            }
            if (m < 4)                                       // kra P-47: one store, lane m == e
                p.ddt_part4[(((size_t)hh * kD + kc + 4 * q + m) * p.NT + i) * 4 + A] = vsel;""")
# ---- epilogue: the tail kernel's row mapping on this wave's own rows
a = t.index("        // l2norm backward: dx = dy*rstd - (dy.y) y rstd  (row dot over the 4 lanes of the row)")
b = t.index("    } else {\n        if (q == 0 && rok) p.db[(size_t)(tok0 + row) * H + hh] = db;\n    }\n}", a)
new_epi = """        // kra P-47: l2norm backward with the bwd_tail kernel's row mapping and operation order on
        // this wave's OWN 16 rows (its stores above): lane l = columns 2l, 2l+1, one 512 B row per
        // instruction, xor-tree row dot -> dq/dk bit-identical to the unfused tail
        __builtin_amdgcn_s_waitcnt(0);
        {
            const int c2 = l * 2;
            #pragma unroll 1
            for (int rb = 0; rb < 16; rb += 4) {
                const int r0 = 16 * A + rb;
                if (!g2_row_ok<M>(r0, cl)) break;            // wave-uniform; rows grow
                float2 dyv[2][4]; short2 yv[2][4]; float r4[2][4];
                #pragma unroll
                for (int w2 = 0; w2 < 2; ++w2) {
                    const float* dy = w2 ? dk_ : dq_;
                    const bf16* y = w2 ? k_ : q_;
                    const float* rs = w2 ? p.k_rstd : p.q_rstd;
                    #pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        if (g2_row_ok<M>(r0 + u, cl)) {
                            dyv[w2][u] = *reinterpret_cast<const float2*>(dy + (r0 + u) * R + c2);
                            yv[w2][u] = *reinterpret_cast<const short2*>(y + (r0 + u) * R + c2);
                            r4[w2][u] = rs[(size_t)(tok0 + r0 + u) * H + hh];
                        } else {
                            dyv[w2][u] = float2{0.f, 0.f}; yv[w2][u] = short2{0, 0}; r4[w2][u] = 0.f;
                        }
                    }
                }
                #pragma unroll
                for (int w2 = 0; w2 < 2; ++w2) {
                    short* out = (w2 ? p.dko : p.dqo) + cb;
                    #pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        const float y0 = bf2f(yv[w2][u].x), y1 = bf2f(yv[w2][u].y);
                        float dot = dyv[w2][u].x * y0 + dyv[w2][u].y * y1;
                        #pragma unroll
                        for (int off = 1; off < 64; off <<= 1)
                            dot += __shfl_xor(dot, off, 64);    // fixed-order tree (the tail's)
                        short2 o;
                        o.x = bf_rne(dyv[w2][u].x * r4[w2][u] - dot * y0 * r4[w2][u]);
                        o.y = bf_rne(dyv[w2][u].y * r4[w2][u] - dot * y1 * r4[w2][u]);
                        if (g2_row_ok<M>(r0 + u, cl))
                            *reinterpret_cast<short2*>(out + (r0 + u) * R + c2) = o;
                    }
                }
            }
        }
"""
t = t[:a] + new_epi + t[b:]
P.write_text(t)
print("P-47 (B4b-lite) APPLIED")
