"""kra P-75 probe (bitwise, store placement only): spread the TP8 dhu's HbN (bN(dh)) store burst.

P-74 SQTT (results/kda-gfx936-20261016-p74-dhu-ka): right after barrier A all 8 waves of the CTA issue their 2 HbN
stores (16 x 512 B = 8 KB; a store costs ~2.6x a read in TA cycles), and the step's largest critical item is a later
store that waits ~520-560 cycles to issue behind that burst.  The bN values are still gathered from dhB right after
barrier A; only the two stores move.  Template int SS on g2_dhu_v8b_kernel, selected by KDA_G2_DHU_SS for the
production LIN instance when 4*nseg*H <= 80 (probe):
  1  all waves store after GEMM-A (before the KgA prefetch)
  2  v1 = 1 waves store after GEMM-A, v1 = 0 waves keep the production position (burst halved)
  3  each wave interleaves: vt 0 after GEMM-A kt 3, vt 1 after kt 7 (sched_barrier-pinned)
    python3 p75_dhu_spread.py <tree>"""
import sys
from pathlib import Path

f = Path(sys.argv[1]) / "csrc" / "g2" / "g2_engine.cuh"
s = f.read_text()


def sub(b, old, new, cnt=1):
    assert b.count(old) == cnt, (old[:80], b.count(old))
    return b.replace(old, new)


s = sub(s, """template <int MODE = 0, bool LIN = false>
__global__ void __launch_bounds__(512) g2_dhu_v8b_kernel(BwdNParams p) {""",
        """template <int MODE = 0, bool LIN = false, int SS = 0>   // kra P-75: SS
__global__ void __launch_bounds__(512) g2_dhu_v8b_kernel(BwdNParams p) {""")
head, rest = s.split("__global__ void __launch_bounds__(512) g2_dhu_v8b_kernel(BwdNParams p) {", 1)
body, tail = rest.split("// ================================================================ R1a-dhu: 4-wave form", 1)
old = """        if (hasBN) {
            const int q4_ = l >> 4, m_ = l & 15;
            #pragma unroll
            for (int vt = 0; vt < 2; ++vt) {
                v4bh val;
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    val[e] = reinterpret_cast<const short*>(dhB)[(16 * vt + 4 * q4_ + e) * kPadK + 16 * wv + pi16(m_)];
                const size_t off = ((((size_t)i * p.H + bh) * 8 + wv) * 8 + (2 * vs + vt)) * 256 + l * 4;
                *const_cast<v4bh*>(reinterpret_cast<const v4bh*>(p.HbN + off)) = val;
            }
        }
        f4 acc = f4{0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt)
            acc = mmac(ka[kt], ld8(dhB + (16 * v1 + (l & 15)) * kPadK + 16 * kt + (l >> 4) * 4), acc);
        loadKA(nxt);"""
new = """        v4bh bnv0, bnv1;   // kra P-75: SS > 0 stores them later
        auto bn_off = [&](int vt) { return ((((size_t)i * p.H + bh) * 8 + wv) * 8 + (2 * vs + vt)) * 256 + l * 4; };
        auto bn_st = [&](int vt, const v4bh& v) { *const_cast<v4bh*>(reinterpret_cast<const v4bh*>(p.HbN + bn_off(vt))) = v; };
        if (hasBN) {
            const int q4_ = l >> 4, m_ = l & 15;
            #pragma unroll
            for (int vt = 0; vt < 2; ++vt) {
                v4bh val;
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    val[e] = reinterpret_cast<const short*>(dhB)[(16 * vt + 4 * q4_ + e) * kPadK + 16 * wv + pi16(m_)];
                if (vt == 0) bnv0 = val; else bnv1 = val;
            }
            if constexpr (SS == 0) { bn_st(0, bnv0); bn_st(1, bnv1); }
            if constexpr (SS == 2) { if (v1 == 0) { bn_st(0, bnv0); bn_st(1, bnv1); } }
        }
        f4 acc = f4{0.f, 0.f, 0.f, 0.f};
        if constexpr (SS == 3) {
            #pragma unroll
            for (int kt = 0; kt < 8; ++kt) {
                acc = mmac(ka[kt], ld8(dhB + (16 * v1 + (l & 15)) * kPadK + 16 * kt + (l >> 4) * 4), acc);
                if (hasBN && (kt == 3 || kt == 7)) {
                    __builtin_amdgcn_sched_barrier(0);
                    if (kt == 3) bn_st(0, bnv0); else bn_st(1, bnv1);
                    __builtin_amdgcn_sched_barrier(0);
                }
            }
        } else {
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt)
            acc = mmac(ka[kt], ld8(dhB + (16 * v1 + (l & 15)) * kPadK + 16 * kt + (l >> 4) * 4), acc);
        }
        if constexpr (SS == 1) if (hasBN) { __builtin_amdgcn_sched_barrier(0); bn_st(0, bnv0); bn_st(1, bnv1); __builtin_amdgcn_sched_barrier(0); }
        if constexpr (SS == 2) if (hasBN && v1 != 0) { __builtin_amdgcn_sched_barrier(0); bn_st(0, bnv0); bn_st(1, bnv1); __builtin_amdgcn_sched_barrier(0); }
        loadKA(nxt);"""
body = sub(body, old, new)
s = head + "__global__ void __launch_bounds__(512) g2_dhu_v8b_kernel(BwdNParams p) {" + body + \
    "// ================================================================ R1a-dhu: 4-wave form" + tail
s = sub(s, """        if (prod && lin) hipLaunchKernelGGL((g2_dhu_v8b_kernel<1, true>), dim3(kK / kBV, nseg * H), dim3(512), dyn((g2_dhu_v8b_kernel<1, true>)), s, p);""",
        """        const long ss = ctas <= 80 ? g2_env_long("KDA_G2_DHU_SS", 0) : 0;   // kra P-75 probe
        if (prod && lin && ss == 1) hipLaunchKernelGGL((g2_dhu_v8b_kernel<1, true, 1>), dim3(kK / kBV, nseg * H), dim3(512), dyn((g2_dhu_v8b_kernel<1, true, 1>)), s, p);
        else if (prod && lin && ss == 2) hipLaunchKernelGGL((g2_dhu_v8b_kernel<1, true, 2>), dim3(kK / kBV, nseg * H), dim3(512), dyn((g2_dhu_v8b_kernel<1, true, 2>)), s, p);
        else if (prod && lin && ss == 3) hipLaunchKernelGGL((g2_dhu_v8b_kernel<1, true, 3>), dim3(kK / kBV, nseg * H), dim3(512), dyn((g2_dhu_v8b_kernel<1, true, 3>)), s, p);
        else if (prod && lin) hipLaunchKernelGGL((g2_dhu_v8b_kernel<1, true>), dim3(kK / kBV, nseg * H), dim3(512), dyn((g2_dhu_v8b_kernel<1, true>)), s, p);""")
f.write_text(s)
print("P-75 APPLIED")
