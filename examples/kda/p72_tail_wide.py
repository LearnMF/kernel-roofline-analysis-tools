"""kra P-72 probe (bitwise): bwd_tail phase B (l2norm backward of q and k) with 2 rows per wave instruction.

Plan-3 attribution (results/kda-gfx936-20261015-plan3): bwd_tail is VMEM-issue bound -- 39% of its wave-cycles are
VMEM instructions waiting to issue (FLAT_RD 30.6%, FLAT_WR 8.3%); phase B issues per wave 32 dwordx2 (dy) + 32 dword
(y) loads + 32 dword (short2) stores.  New lane mapping: lanes 0-31 take row r, lanes 32-63 row r+1, each lane 4
adjacent columns -> per row pair one dwordx4 dy load, one dwordx2 y load, one dwordx2 store (half the VMEM).
The row dot keeps the production fixed-order tree exactly: production lane l holds p_l = dy_2l*y_2l + dy_2l+1*y_2l+1,
level off=1 forms p_2j + p_2j+1 (= this lane's pa + pb), levels off=2..32 pair the 32 groups by index bits -- here
lanes j = 0..31 of each half with off = 1..16.  Float addition is commutative, every operand pair is the same ->
bit-identical.
    python3 p72_tail_wide.py <tree>"""
import sys
from pathlib import Path

f = Path(sys.argv[1]) / "csrc" / "g2" / "g2_bwd_tail.cuh"
s = f.read_text()
old = """            // q and k interleaved at the 4-row batch level: 8 row-loads in flight
            for (int rb = 0; rb < 16; rb += 4) {"""
new = """            // q and k interleaved at the 4-row batch level: 8 row-loads in flight
            if (__builtin_amdgcn_readfirstlane(p.NT) > 0 && g2_tail_wide()) {   // kra P-72: 2 rows / instruction
            const int hl = l >> 5, c4 = (l & 31) * 4;
            for (int rb = 0; rb < 16; rb += 4) {
                const int r0 = wv * 16 + rb;
                if (!g2_row_ok<M>(r0, cl)) continue;       // wave-uniform; rows r0..r0+3
                float4 dyv[2][2]; v4bh yv[2][2]; float rr[2][2]; bool okr[2];
                #pragma unroll
                for (int up = 0; up < 2; ++up) okr[up] = g2_row_ok<M>(r0 + 2 * up + hl, cl);
                #pragma unroll
                for (int w2 = 0; w2 < 2; ++w2) {
                    const float* dy = w2 ? p.dk : p.dq;
                    const short* y = w2 ? p.kn : p.qn;
                    #pragma unroll
                    for (int up = 0; up < 2; ++up) {
                        const size_t so = (size_t)(t0 + r0 + 2 * up + hl) * p.H * kK + hh * kK;
                        if (okr[up]) {
                            dyv[w2][up] = *reinterpret_cast<const float4*>(dy + so + c4);
                            yv[w2][up] = *reinterpret_cast<const v4bh*>(y + so + c4);
                        } else {
                            dyv[w2][up] = float4{0.f, 0.f, 0.f, 0.f}; yv[w2][up] = v4bh{0, 0, 0, 0};
                        }
                        const float ra = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(rsl[w2]), rb + 2 * up));
                        const float rb_ = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(rsl[w2]), rb + 2 * up + 1));
                        rr[w2][up] = okr[up] ? (hl ? rb_ : ra) : 0.f;
                    }
                }
                #pragma unroll
                for (int w2 = 0; w2 < 2; ++w2) {
                    short* out = w2 ? p.dko : p.dqo;
                    #pragma unroll
                    for (int up = 0; up < 2; ++up) {
                        const float y0 = bf2f(yv[w2][up][0]), y1 = bf2f(yv[w2][up][1]);
                        const float y2 = bf2f(yv[w2][up][2]), y3 = bf2f(yv[w2][up][3]);
                        const float4 d = dyv[w2][up];
                        const float pa = d.x * y0 + d.y * y1;
                        const float pb = d.z * y2 + d.w * y3;
                        float dot = pa + pb;                    // = production level off=1
                        #pragma unroll
                        for (int off = 1; off < 32; off <<= 1)
                            dot += __shfl_xor(dot, off, 64);    // = production levels off=2..32
                        const float r = rr[w2][up];
                        v4bh o;
                        o[0] = bf_rne(d.x * r - dot * y0 * r);
                        o[1] = bf_rne(d.y * r - dot * y1 * r);
                        o[2] = bf_rne(d.z * r - dot * y2 * r);
                        o[3] = bf_rne(d.w * r - dot * y3 * r);
                        if (okr[up])
                            *reinterpret_cast<v4bh*>(out + (size_t)(t0 + r0 + 2 * up + hl) * p.H * kK + hh * kK + c4) = o;
                    }
                }
            }
            } else
            for (int rb = 0; rb < 16; rb += 4) {"""
assert s.count(old) == 1
s = s.replace(old, new)
old2 = """template <bool VL>
__global__ void __launch_bounds__(256) g2_bwd_tail_kernel(TailParams p) {"""
assert s.count(old2) == 1
s = s.replace(old2, """__device__ __forceinline__ bool g2_tail_wide() { return true; }   // kra P-72 (probe: always on)
""" + old2)
f.write_text(s)
print("P-72 APPLIED")
