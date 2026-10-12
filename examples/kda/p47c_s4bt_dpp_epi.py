"""kra P-47c (on top of P-47 [+P-47b]): the two costs SQTT (8K/12) found in the fused s4b+tail.
  * epilogue (~26K cycles/wave): no full s_waitcnt(0) drain (same-wave store->load to the same address
    is ordered by the hardware), and all 16 rows' loads issued at once (one wait instead of four)
  * in-loop dg (~48K cycles/wave): the reverse suffix over the 16 rows and the ddt column sums use DPP
    row_shl adds (VALU, no LDS-pipe bpermute round trips); the column total is lane m==0's suffix, so
    the ddt partials are stored as ONE 16 B store per kt by the 4 lanes m==0, in a new layout
    ddt_part4[h][chunk][rowblock][128] (columns contiguous); the reduce kernel follows the layout.
python3 p47c_s4bt_dpp_epi.py <tree with P-47>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wy.cuh"
t = P.read_text()


def rep(old, new):
    global t
    assert t.count(old) == 1, (old[:90], t.count(old))
    t = t.replace(old, new)


# ---- DPP helper (row_shl: lane m receives lane m+off of its 16-lane row; out of row -> 0)
rep("template <int A, int M, bool TAIL = false>\n__device__ __forceinline__ void s4b_tail(",
    """template <int OFF>
__device__ __forceinline__ float dpp_shl_add(float x) {      // kra P-47c: x + x[lane m+OFF] (0 past the row)
    const int y = __builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x100 + OFF, 0xF, 0xF, true);
    return x + __int_as_float(y);
}
__device__ __forceinline__ float row_suffix16(float x) {    // inclusive suffix over the 16-lane row
    x = dpp_shl_add<1>(x); x = dpp_shl_add<2>(x); x = dpp_shl_add<4>(x); x = dpp_shl_add<8>(x);
    return x;
}

template <int A, int M, bool TAIL = false>
__device__ __forceinline__ void s4b_tail(""")
rep("""                float x = rok ? og[e] : 0.f;                 // rows past the chunk end contribute 0
                #pragma unroll
                for (int off = 1; off < 16; off <<= 1) {    // reverse inclusive suffix over the 16 rows
                    const float y = __shfl_down(x, off, 16);
                    x += (m + off < 16) ? y : 0.f;
                }
                sx[e] = x;""", """                sx[e] = row_suffix16(rok ? og[e] : 0.f);   // kra P-47c: DPP suffix over the 16 rows""")
rep("""            float vsel = 0.f;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {                    // ddt column partial over the wave's rows
                float v = cv[e];
                v += __shfl_xor(v, 1, 16); v += __shfl_xor(v, 2, 16);
                v += __shfl_xor(v, 4, 16); v += __shfl_xor(v, 8, 16);
                vsel = (m == e) ? v : vsel;
            }
            if (m < 4)                                       // kra P-47: one store, lane m == e
                p.ddt_part4[(((size_t)hh * kD + kc + 4 * q + m) * p.NT + i) * 4 + A] = vsel;""",
    """            f4 csum;                                         // kra P-47c: column totals = lane m==0's suffix
            #pragma unroll
            for (int e = 0; e < 4; ++e) csum[e] = row_suffix16(cv[e]);
            if (m == 0)                                      // ddt_part4[h][chunk][A][128], one 16 B store
                *reinterpret_cast<f4*>(p.ddt_part4 + (((size_t)hh * p.NT + i) * 4 + A) * kD + kc + 4 * q) = csum;""")
# ---- reduce kernel for the new layout
rep("""    if (tid < kK) {
        const size_t base = ((size_t)hh * kK + tid) * NT * 4;
        float s = 0.f, c = 0.f;
        for (int k = 0; k < NT * 4; ++k) {
            const float y = ddt_part4[base + k] - c, tt = s + y;
            c = (tt - s) - y; s = tt;
        }""", """    if (tid < kK) {
        const size_t base = (size_t)hh * NT * 4 * kK + tid;     // kra P-47c: [h][chunk][A][128]
        float s = 0.f, c = 0.f;
        for (int k = 0; k < NT * 4; ++k) {
            const float y = ddt_part4[base + (size_t)k * kK] - c, tt = s + y;
            c = (tt - s) - y; s = tt;
        }""")
# ---- epilogue: no drain, all 16 rows at once
a = t.index("        __builtin_amdgcn_s_waitcnt(0);\n        {\n            const int c2 = l * 2;")
b = t.index("    } else {\n        if (q == 0 && rok) p.db[(size_t)(tok0 + row) * H + hh] = db;\n    }\n}", a)
t = t[:a] + """        {   // kra P-47c: all 16 rows of this wave in flight at once; same-wave RAW needs no drain
            const int c2 = l * 2;
            float2 dyv[2][16]; short2 yv[2][16]; float r4[2][16];
            #pragma unroll
            for (int w2 = 0; w2 < 2; ++w2) {
                const float* dy = w2 ? dk_ : dq_;
                const bf16* y = w2 ? k_ : q_;
                const float* rs = w2 ? p.k_rstd : p.q_rstd;
                #pragma unroll
                for (int u = 0; u < 16; ++u) {
                    const int rr = 16 * A + u;
                    if (g2_row_ok<M>(rr, cl)) {
                        dyv[w2][u] = *reinterpret_cast<const float2*>(dy + rr * R + c2);
                        yv[w2][u] = *reinterpret_cast<const short2*>(y + rr * R + c2);
                        r4[w2][u] = rs[(size_t)(tok0 + rr) * H + hh];
                    } else {
                        dyv[w2][u] = float2{0.f, 0.f}; yv[w2][u] = short2{0, 0}; r4[w2][u] = 0.f;
                    }
                }
            }
            #pragma unroll
            for (int w2 = 0; w2 < 2; ++w2) {
                short* out = (w2 ? p.dko : p.dqo) + cb;
                #pragma unroll
                for (int u = 0; u < 16; ++u) {
                    const int rr = 16 * A + u;
                    const float y0 = bf2f(yv[w2][u].x), y1 = bf2f(yv[w2][u].y);
                    float dot = dyv[w2][u].x * y0 + dyv[w2][u].y * y1;
                    #pragma unroll
                    for (int off = 1; off < 64; off <<= 1)
                        dot += __shfl_xor(dot, off, 64);        // fixed-order tree (the tail's)
                    short2 o;
                    o.x = bf_rne(dyv[w2][u].x * r4[w2][u] - dot * y0 * r4[w2][u]);
                    o.y = bf_rne(dyv[w2][u].y * r4[w2][u] - dot * y1 * r4[w2][u]);
                    if (g2_row_ok<M>(rr, cl)) *reinterpret_cast<short2*>(out + rr * R + c2) = o;
                }
            }
        }
""" + t[b:]
P.write_text(t)
print("P-47c APPLIED")
