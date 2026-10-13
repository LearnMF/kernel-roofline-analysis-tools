"""kra P-54 (A-class, bwd_tail): fewer, wider VMEM instructions in the TA-bound backward tail
(ceiling cards: TA 0.92-0.99, VMEM instructions 4x the ideal; 10% of the op at every shape).

  a  phase B: q_rstd / k_rstd are read ONCE per warp (lane j of the warp's 16-row block loads row j)
     and broadcast per row with v_readlane -- the per-row uniform-address VMEM load (32 per lane) goes;
     phase A: g_raw is read as one short2 per row for the lane's two adjacent columns (16 loads of
     4 B instead of 32 of 2 B); the two per-column gate blocks stay separate (production fma form).
  b  phase B (l2norm bwd): 4 adjacent columns per lane, the two half-waves take a row pair -> float4 dy,
     short4 y, short4 out (half the VMEM instructions), and the row dot runs as the in-lane pair add +
     a 5-level butterfly within 32 lanes: the production tree exactly (its level 1, lane l + lane l^1,
     is the in-lane add of the same two pair sums; levels xor 2..32 map to xor 1..16).
  c  wv = readfirstlane(tid >> 6): the compiler then knows every row index is wave-uniform and moves the
     per-row address math to SALU / SGPR bases (SQTT: valu_int_addr -> global_load is the tail's top bubble).
Same values, same arithmetic order -> bit-identical.
    python3 p54_tail_vmem.py <tree> [a|ab|ac|abc]"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_bwd_tail.cuh"
steps = sys.argv[2] if len(sys.argv) > 2 else "a"
t = P.read_text()


def sub(old, new, cnt=1):
    global t
    assert t.count(old) == cnt, (old[:90], t.count(old))
    t = t.replace(old, new)


B_NEW = """            // kra P-54b: 4 adjacent columns per lane; half-wave h of the warp takes row 2k + h of the row
            // pair k (float4 dy / short4 y / short4 out: half the VMEM instructions).  The row dot is the
            // production tree: in-lane pA + pB == its level 1 (lane l + lane l^1), then xor 1..16 within the
            // half-wave == its levels xor 2..32.  rstd: lane j < 16 holds row 16w + j (v_readlane per row).
            const int half = l >> 5, c4 = 4 * (l & 31);
            float rsl[2];
            {
                const int jr = wv * 16 + (l & 15);
                const bool okr = g2_row_ok<M>(jr, cl);
                rsl[0] = okr ? p.q_rstd[(size_t)(t0 + jr) * p.H + hh] : 0.f;
                rsl[1] = okr ? p.k_rstd[(size_t)(t0 + jr) * p.H + hh] : 0.f;
            }
            for (int rb = 0; rb < 16; rb += 4) {            // 4 rows = 2 row pairs per batch
                const int r0 = wv * 16 + rb;
                if (!g2_row_ok<M>(r0, cl)) continue;       // wave-uniform; rows r0..r0+3
                float4 dyv[2][2]; short4 yv[2][2]; float r4[2][2];
                #pragma unroll
                for (int w2 = 0; w2 < 2; ++w2) {
                    const float* dy = w2 ? p.dk : p.dq;
                    const short* y = w2 ? p.kn : p.qn;
                    #pragma unroll
                    for (int pr = 0; pr < 2; ++pr) {
                        const int rr = r0 + 2 * pr + half;  // this half-wave's row
                        const float ra = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(rsl[w2]), rb + 2 * pr));
                        const float rv = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(rsl[w2]), rb + 2 * pr + 1));
                        if (g2_row_ok<M>(rr, cl)) {
                            const size_t so = ((size_t)(t0 + rr) * p.H + hh) * kK + c4;
                            dyv[w2][pr] = *reinterpret_cast<const float4*>(dy + so);
                            yv[w2][pr] = *reinterpret_cast<const short4*>(y + so);
                            r4[w2][pr] = half ? rv : ra;
                        } else {
                            dyv[w2][pr] = float4{0.f, 0.f, 0.f, 0.f}; yv[w2][pr] = short4{0, 0, 0, 0}; r4[w2][pr] = 0.f;
                        }
                    }
                }
                #pragma unroll
                for (int w2 = 0; w2 < 2; ++w2) {
                    short* out = w2 ? p.dko : p.dqo;
                    #pragma unroll
                    for (int pr = 0; pr < 2; ++pr) {
                        const int rr = r0 + 2 * pr + half;
                        const float4 d = dyv[w2][pr];
                        const float r = r4[w2][pr];
                        const float y0 = bf2f(yv[w2][pr].x), y1 = bf2f(yv[w2][pr].y);
                        const float y2 = bf2f(yv[w2][pr].z), y3 = bf2f(yv[w2][pr].w);
                        short4 o;
                        {
                        // the production instruction forms, pinned: pair sum = v_pk_mul + v_add (UNfused);
                        // out = fma(dy, r, -((dot * y) * r)) (v_pk_mul x2 + v_pk_fma)
                        #pragma clang fp contract(off)
                        const float pA = d.x * y0 + d.y * y1;           // production lane 2(l&31)
                        const float pB = d.z * y2 + d.w * y3;           // production lane 2(l&31)+1
                        float dot = pA + pB;                            // production level 1 (xor 1)
                        #pragma unroll
                        for (int off = 1; off < 32; off <<= 1)
                            dot += __shfl_xor(dot, off, 32);            // production levels xor 2..32
                        o.x = bf_rne(__builtin_fmaf(d.x, r, -((dot * y0) * r)));
                        o.y = bf_rne(__builtin_fmaf(d.y, r, -((dot * y1) * r)));
                        o.z = bf_rne(__builtin_fmaf(d.z, r, -((dot * y2) * r)));
                        o.w = bf_rne(__builtin_fmaf(d.w, r, -((dot * y3) * r)));
                        }
                        if (g2_row_ok<M>(rr, cl))
                            *reinterpret_cast<short4*>(out + ((size_t)(t0 + rr) * p.H + hh) * kK + c4) = o;
                    }
                }
            }
"""

if "a" in steps:
    # ---- phase A: g_raw as short2 per row (both columns), loaded once before the column blocks
    sub("""            short b0[16], b1[16];""",
        """            short b0[16], b1[16];
            short2 gr2[16];                                 // kra P-54a: g_raw of both columns, 4 B per row
            #pragma unroll
            for (int u = 0; u < 16; ++u) {
                const int t = tb + 15 - u;
                gr2[u] = *reinterpret_cast<const short2*>(
                    p.g_raw + ((size_t)(t0 + g2_row_g<M>(t, cl)) * p.H + hh) * kK + d0);
            }""")
    sub("""                    const size_t sog = ((size_t)(t0 + g2_row_g<M>(t, cl)) * p.H + hh) * kK + d0 + j;
                    const float suffix = loc[u] + carry;
                    const float gg = bf2f(p.g_raw[sog]) + dtbj;""",
        """                    const float suffix = loc[u] + carry;
                    const float gg = bf2f(j ? gr2[u].y : gr2[u].x) + dtbj;   // kra P-54a""")
if "b" in steps:
    a0 = t.index("            // q and k interleaved at the 4-row batch level: 8 row-loads in flight\n")
    a1 = t.index("        }\n        if (p.use_beta && tid < kBT && g2_row_ok<M>(tid, cl)) {")
    t = t[:a0] + B_NEW + t[a1:]
elif "a" in steps:
    # ---- phase B: rstd once per warp + readlane broadcast
    sub("""            // q and k interleaved at the 4-row batch level: 8 row-loads in flight
            for (int rb = 0; rb < 16; rb += 4) {""",
        """            // kra P-54a: lane j (j < 16) of the warp holds rstd of row 16w + j; rows are broadcast with
            // v_readlane (the per-row uniform-address VMEM loads go).  Rows past the end hold 0, as before.
            float rsl[2];
            {
                const int jr = wv * 16 + (l & 15);
                const bool okr = g2_row_ok<M>(jr, cl);
                rsl[0] = okr ? p.q_rstd[(size_t)(t0 + jr) * p.H + hh] : 0.f;
                rsl[1] = okr ? p.k_rstd[(size_t)(t0 + jr) * p.H + hh] : 0.f;
            }
            // q and k interleaved at the 4-row batch level: 8 row-loads in flight
            for (int rb = 0; rb < 16; rb += 4) {""")
    sub("""                    const float* rs = w2 ? p.k_rstd : p.q_rstd;
""", "")
    sub("""                            r4[w2][u] = rs[(size_t)(row + u) * p.H + hh];""",
        """                            r4[w2][u] = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(rsl[w2]), rb + u));""")
if "c" in steps:
    sub("""    const int i = blockIdx.x, hh = blockIdx.y, tid = threadIdx.x;
    const int wv = tid >> 6, l = tid & 63;              // wave64
    const long t0 = g2_row_tok<VL>(p.ctok, i);          // chunk row 0 (packed token)
    __shared__ float sDA[8 * 64];""",
        """    const int i = blockIdx.x, hh = blockIdx.y, tid = threadIdx.x;
    const int wv = __builtin_amdgcn_readfirstlane(tid >> 6), l = tid & 63;   // wave64; kra P-54c: uniform
    const long t0 = g2_row_tok<VL>(p.ctok, i);          // chunk row 0 (packed token)
    __shared__ float sDA[8 * 64];""")
P.write_text(t)
print("P-54", steps, "APPLIED")
