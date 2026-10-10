"""Project C probe 4 (compile-only; values NOT validated): the dg-path fusion (scheme B) inside
s4b's kt epilogue -- per kt, the 16 rows of a wave's block run the bwd_tail Phase-A chain
(reverse Kahan suffix over rows 15..0, broadcast by shuffles: every lane of a q-group runs the
chain, lane m keeps step 15-m), block totals via a 256 B LDS array, carry from the blocks
above, gate backward (expf / sigmoid), bf16 dg_raw store and per-column Kahan ddt / dA partials
written to a global scratch -- the fp32 dg store is dropped.  Question: does the epilogue fit
in 256 VGPRs without scratch, and how many instructions does it add per kt?
Usage: python3 c_probe_s4b_dgpath.py <g2 dir>"""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "g2_wy.cuh"
t = p.read_text()
# params: the gate inputs + outputs the fused epilogue needs (probe: appended, unused by host)
s0 = t.index("struct WyS4BParams {")
s1 = t.index("};", s0)
t = t[:s1] + ("    const short* g_raw; const float* A_log; const float* dt_bias; float lower_bound;\n"
              "    short* dg_raw; float* ddt_scr; float* dA_scr;   // C probe 4\n") + t[s1:]
t = t.replace("    float gM[4][16];\n    float cs1[2][4][16], cs2[2][4][16];",
              "    float gM[4][16];\n    float cs1[2][4][16], cs2[2][4][16];\n    float blk[4][16];   // C probe 4: block totals", 1)
a = t.index("__device__ __forceinline__ void s4b_tail(")
b = t.index("\n}\n", a)
body = t[a:b]
old = """        if (rok) {
            *reinterpret_cast<f4*>(dq_ + ro + kc) = oq;
            *reinterpret_cast<f4*>(dk_ + ro + kc) = ok;
            *reinterpret_cast<f4*>(dg_ + ro + kc) = og;
        }"""
assert body.count(old) == 1
new = """        if (rok) {
            *reinterpret_cast<f4*>(dq_ + ro + kc) = oq;
            *reinterpret_cast<f4*>(dk_ + ro + kc) = ok;
        }
        {   // C probe 4: tail Phase A for this kt tile (rows of block A, columns kc+4q..+3)
            float loc[4], run[4] = {0.f, 0.f, 0.f, 0.f}, runC[4] = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int u = 0; u < 16; ++u) {                      // rows 15..0 of the block
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float y = __shfl(og[e], 16 * q + (15 - u), 64) - runC[e];
                    const float tt = run[e] + y;
                    runC[e] = (tt - run[e]) - y;  run[e] = tt;
                    if (m == 15 - u) loc[e] = run[e];
                }
            }
            if (m == 0) {
                #pragma unroll
                for (int e = 0; e < 4; ++e) sm.blk[A][4 * q + e] = run[e];
            }
            lds_barrier();
            const float Ag = expf(p.A_log[hh]);
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int d = kc + 4 * q + e;
                float carry = 0.f, carryC = 0.f;
                for (int bb = A + 1; bb < 4; ++bb) {
                    const float y = sm.blk[bb][4 * q + e] - carryC, tt = carry + y;
                    carryC = (tt - carry) - y;  carry = tt;
                }
                const float suffix = loc[e] + carry;
                const float gg = bf2f(p.g_raw[(size_t)cb + rog + kc - 4 * q + 4 * q + e]) + p.dt_bias[hh * kD + d];
                const float sg = 1.f / (1.f + expf(-(Ag * gg)));
                float bdg = suffix * (p.lower_bound * (sg * (1.f - sg)));
                bdg *= Ag;
                const short b16 = bf_rne(bdg);
                if (rok) p.dg_raw[(size_t)cb + ro + kc + e] = b16;
                float dAs = 0.f, dAsC = 0.f, ddt = 0.f, ddtC = 0.f;   // column chains over the rows
                #pragma unroll
                for (int u = 0; u < 16; ++u) {
                    const float bv = __shfl(bdg, 16 * q + (15 - u), 64);
                    const float gv2 = __shfl(gg, 16 * q + (15 - u), 64);
                    { const float y = bv * gv2 - dAsC, tt = dAs + y; dAsC = (tt - dAs) - y; dAs = tt; }
                    { const float y = bf2f(bf_rne(bv)) - ddtC, tt = ddt + y; ddtC = (tt - ddt) - y; ddt = tt; }
                }
                if (m == 0) {
                    p.ddt_scr[((size_t)(i * H + hh) * 4 + A) * kD + d] = ddt;
                    p.dA_scr[((size_t)(i * H + hh) * 4 + A) * kD + d] = dAs;
                }
            }
        }"""
body = body.replace(old, new)
t = t[:a] + body + t[b:]
p.write_text(t)
print("C PROBE 4 APPLIED")
