"""kra P-48 (A-class, bwd_tail): phase A (dg reverse chunk-cumsum + gate backward) with ADJACENT column
pairs per lane.  The production mapping gives lane l the columns l and l+64, so every access to the
bf16 g_raw / dg_raw is 2 B per lane (128 B per wave instruction) and to the fp32 dg 4 B per lane --
bwd_tail issues 14.2 VMEM instructions per token*head at 225 B each, the most and smallest of all KDA
kernels (TA 65%).  Lane l now owns columns 2l, 2l+1: float2 / short2 accesses, half the instructions.
Per column the arithmetic is unchanged (same Kahan order over the rows, same carry order over the
blocks, same gate expressions), and every per-(column, warp) partial is stored at the production
(c, l) slot of its column, so the ddt column merge and the dA CTA sum run in the production order:
bit-identical.  python3 p48_tail_pairs.py <tree>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_bwd_tail.cuh"
t = P.read_text()
a = t.index("""        } else {
            const int tb = wv * 16;                         // block's first row
            const float A = expf(p.A_log[hh]);""")
b = t.index("""        __syncthreads();
        if (!p.gate_none && tid < kK) {""", a)
new = """        } else {
            const int tb = wv * 16;                         // block's first row
            const float A = expf(p.A_log[hh]);
            // kra P-48: lane l owns the ADJACENT columns d0 = 2l, d0 + 1 (float2 / short2 accesses);
            // partials go to the production (c, l) slot of their column -> production merge order
            const int d0 = 2 * l;
            auto slot = [&](int col, int b_) { return ((col >> 6) * 4 + b_) * 64 + (col & 63); };
            const float2 dtb = *reinterpret_cast<const float2*>(p.dt_bias + (size_t)hh * kK + d0);
            float loc0[16], loc1[16], run0 = 0.f, run0C = 0.f, run1 = 0.f, run1C = 0.f;
            #pragma unroll
            for (int u = 0; u < 16; ++u) {
                const int t = tb + 15 - u;
                const float2 v = g2_row_ok<M>(t, cl)
                    ? *reinterpret_cast<const float2*>(p.dg + ((size_t)(t0 + t) * p.H + hh) * kK + d0)
                    : float2{0.f, 0.f};
                {
                    const float y = v.x - run0C;
                    const float tt = run0 + y;
                    run0C = (tt - run0) - y;  run0 = tt;
                    loc0[u] = run0;
                }
                {
                    const float y = v.y - run1C;
                    const float tt = run1 + y;
                    run1C = (tt - run1) - y;  run1 = tt;
                    loc1[u] = run1;
                }
            }
            sBlk[slot(d0, wv)] = run0;                      // block totals
            sBlk[slot(d0 + 1, wv)] = run1;
            __syncthreads();
            float carry0 = 0.f, carry0C = 0.f, carry1 = 0.f, carry1C = 0.f;   // blocks above
            for (int b = wv + 1; b < 4; ++b) {
                {
                    const float y = sBlk[slot(d0, b)] - carry0C, tt = carry0 + y;
                    carry0C = (tt - carry0) - y;  carry0 = tt;
                }
                {
                    const float y = sBlk[slot(d0 + 1, b)] - carry1C, tt = carry1 + y;
                    carry1C = (tt - carry1) - y;  carry1 = tt;
                }
            }
            float dAs0 = 0.f, dAs0C = 0.f, ddt0 = 0.f, ddt0C = 0.f;
            float dAs1 = 0.f, dAs1C = 0.f, ddt1 = 0.f, ddt1C = 0.f;
            short b0[16], b1[16];
            // per column a separate straight-line block (the opaque branch keeps the SLP vectorizer from
            // pairing the two columns into packed fp32 ops, which would un-fuse the production's fmas)
            auto gate_col = [&](const float (&loc)[16], float carry, int j, float dtbj, short (&bo)[16],
                                float& dAs, float& dAsC, float& ddt, float& ddtC) {
                #pragma unroll
                for (int u = 0; u < 16; ++u) {
                    const int t = tb + 15 - u;
                    const size_t sog = ((size_t)(t0 + g2_row_g<M>(t, cl)) * p.H + hh) * kK + d0 + j;
                    const float suffix = loc[u] + carry;
                    const float gg = bf2f(p.g_raw[sog]) + dtbj;
                    const float sg = 1.f / (1.f + expf(-(A * gg)));
                    float bdg = suffix * (p.lower_bound * (sg * (1.f - sg)));   // dyg*t1
                    bdg *= A;
                    {   // dAs += bdg * gg (compensated)
                        const float y = bdg * gg - dAsC, tt = dAs + y;
                        dAsC = (tt - dAs) - y;  dAs = tt;
                    }
                    const short b16 = bf_rne(bdg);          // round to g_raw dtype
                    {   // ddt += bf2f(b16)
                        const float y = bf2f(b16) - ddtC, tt = ddt + y;
                        ddtC = (tt - ddt) - y;  ddt = tt;
                    }
                    bo[u] = b16;
                }
            };
            gate_col(loc0, carry0, 0, dtb.x, b0, dAs0, dAs0C, ddt0, ddt0C);
            if (__builtin_amdgcn_readfirstlane(p.NT) > 0)           // opaque, always true
                gate_col(loc1, carry1, 1, dtb.y, b1, dAs1, dAs1C, ddt1, ddt1C);
            #pragma unroll
            for (int u = 0; u < 16; ++u) {
                const int t = tb + 15 - u;
                if (g2_row_ok<M>(t, cl))
                    *reinterpret_cast<short2*>(p.dg_raw + ((size_t)(t0 + t) * p.H + hh) * kK + d0) = short2{b0[u], b1[u]};
            }
            sDDt[slot(d0, wv)] = ddt0;                      // merged below in the production order
            sDDt[slot(d0 + 1, wv)] = ddt1;
            sDA[slot(d0, wv)] = dAs0;
            sDA[slot(d0 + 1, wv)] = dAs1;
        }
"""
t = t[:a] + new + t[b:]
P.write_text(t)
print("P-48 APPLIED")
