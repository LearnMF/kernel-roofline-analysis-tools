"""kra P-67 probe (bitwise): TP8 (chain-bound) dhu runs the 4-wave g2_dhu_v8b4 form with the forward recurrence's
kernel-local fixes.

Critical paths at TP8 16K/12 (cpdhu_p65_d1 / cpdhu4_p65_d1): the production 8-wave v8b spends 46% of its 3753-cycle
step on VMEM issue stalls (two waves per SIMD duplicate the KgA loads and queue behind each other); the 4-wave
v8b4 is as slow (3718) but issue-bound like the forward step was before P-58/P-65 -- ~300 cycles of 64-bit address
VALU and ~475 cycles of interleaved LDS read waits.  New template flags of g2_dhu_v8b4_kernel:
  SG  wave index through readfirstlane (uniform address parts in SGPRs; P-58);
  KB  every GEMM issues its LDS operand batch (16 dhB / 8 dOs + 8 vB fragments) ahead of its MMAC chain (P-65).
Dispatch: 4 * nseg * H <= KDA_G2_DHU4_TP8_MAX_CTAS (80) -> v8b4<0, lin, true, true>.  Operands and MMAC order
unchanged (v8b4 is bit-identical to v8b) -> bit-identical.
    python3 p67_dhu4_tp8.py <tree>"""
import sys
from pathlib import Path

f = Path(sys.argv[1]) / "csrc" / "g2" / "g2_engine.cuh"
s = f.read_text()


def sub(s, old, new, cnt=1):
    assert s.count(old) == cnt, (old[:80], s.count(old))
    return s.replace(old, new)


s = sub(s, """template <int MODE = 0, bool LIN = false>
__global__ void __launch_bounds__(256) g2_dhu_v8b4_kernel(BwdNParams p) {""",
        """template <int MODE = 0, bool LIN = false, bool SG = false, bool KB = false>   // kra P-67: SG / KB
__global__ void __launch_bounds__(256) g2_dhu_v8b4_kernel(BwdNParams p) {""")
head, rest = s.split("g2_dhu_v8b4_kernel(BwdNParams p) {", 1)
body, tail = rest.split('extern "C" int g2_dhu_w8b(', 1)
body = sub(body, "    const int tid = threadIdx.x, wv = tid >> 6, l = tid & 63;   // wv in 0..3",
           "    const int tid = threadIdx.x, wv = SG ? __builtin_amdgcn_readfirstlane(tid >> 6) : (tid >> 6), l = tid & 63;"
           "   // wv in 0..3; kra P-67")
body = sub(body, """        f4 acc0 = f4{0.f, 0.f, 0.f, 0.f}, acc1 = f4{0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
            acc0 = mmac(ka[kt], ld8(dhB + (16 * 0 + (l & 15)) * kPadK + 16 * kt + (l >> 4) * 4), acc0);
            acc1 = mmac(ka[kt], ld8(dhB + (16 * 1 + (l & 15)) * kPadK + 16 * kt + (l >> 4) * 4), acc1);
        }""", """        f4 acc0 = f4{0.f, 0.f, 0.f, 0.f}, acc1 = f4{0.f, 0.f, 0.f, 0.f};
        if constexpr (KB) {   // kra P-67: the 16 dhB fragments first, then the MMAC chain
            v4bh hf0[8], hf1[8];
            #pragma unroll
            for (int kt = 0; kt < 8; ++kt) {
                hf0[kt] = ld8(dhB + (16 * 0 + (l & 15)) * kPadK + 16 * kt + (l >> 4) * 4);
                hf1[kt] = ld8(dhB + (16 * 1 + (l & 15)) * kPadK + 16 * kt + (l >> 4) * 4);
            }
            __builtin_amdgcn_sched_barrier(0);
            #pragma unroll
            for (int kt = 0; kt < 8; ++kt) {
                acc0 = mmac(ka[kt], hf0[kt], acc0);
                acc1 = mmac(ka[kt], hf1[kt], acc1);
            }
        } else {
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
            acc0 = mmac(ka[kt], ld8(dhB + (16 * 0 + (l & 15)) * kPadK + 16 * kt + (l >> 4) * 4), acc0);
            acc1 = mmac(ka[kt], ld8(dhB + (16 * 1 + (l & 15)) * kPadK + 16 * kt + (l >> 4) * 4), acc1);
        }
        }""")
old_gb = """        #pragma unroll
        for (int vt = 0; vt < 2; ++vt) {
            f4 qd = f4{0.f, 0.f, 0.f, 0.f}, wd = f4{0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int rt = 0; rt < 4; ++rt) {
                qd = mmac(dOs[s & 1][(vt * 4 + rt) * 64 + l], qb0[rt], qd);
                wd = mmac(ld8(vB + (16 * vt + (l & 15)) * kPadT + 16 * rt + (l >> 4) * 4), wb0[rt], wd);
            }
            st0[vt] += qd * p.scale - wd;
        }
        #pragma unroll
        for (int vt = 0; vt < 2; ++vt) {
            f4 qd = f4{0.f, 0.f, 0.f, 0.f}, wd = f4{0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int rt = 0; rt < 4; ++rt) {
                qd = mmac(dOs[s & 1][(vt * 4 + rt) * 64 + l], qb1[rt], qd);
                wd = mmac(ld8(vB + (16 * vt + (l & 15)) * kPadT + 16 * rt + (l >> 4) * 4), wb1[rt], wd);
            }
            st1[vt] += qd * p.scale - wd;
        }"""
new_gb = """        if constexpr (KB) {   // kra P-67: the 8 dOs + 8 vB fragments (shared by both K-blocks) first
            v4bh df[2][4], vf[2][4];
            #pragma unroll
            for (int vt = 0; vt < 2; ++vt)
                #pragma unroll
                for (int rt = 0; rt < 4; ++rt) {
                    df[vt][rt] = dOs[s & 1][(vt * 4 + rt) * 64 + l];
                    vf[vt][rt] = ld8(vB + (16 * vt + (l & 15)) * kPadT + 16 * rt + (l >> 4) * 4);
                }
            __builtin_amdgcn_sched_barrier(0);
            #pragma unroll
            for (int vt = 0; vt < 2; ++vt) {
                f4 qd = f4{0.f, 0.f, 0.f, 0.f}, wd = f4{0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int rt = 0; rt < 4; ++rt) {
                    qd = mmac(df[vt][rt], qb0[rt], qd);
                    wd = mmac(vf[vt][rt], wb0[rt], wd);
                }
                st0[vt] += qd * p.scale - wd;
            }
            #pragma unroll
            for (int vt = 0; vt < 2; ++vt) {
                f4 qd = f4{0.f, 0.f, 0.f, 0.f}, wd = f4{0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int rt = 0; rt < 4; ++rt) {
                    qd = mmac(df[vt][rt], qb1[rt], qd);
                    wd = mmac(vf[vt][rt], wb1[rt], wd);
                }
                st1[vt] += qd * p.scale - wd;
            }
        } else {
""" + old_gb + "\n        }"
body = sub(body, old_gb, new_gb)
s = head + "g2_dhu_v8b4_kernel(BwdNParams p) {" + body + 'extern "C" int g2_dhu_w8b(' + tail
s = sub(s, """    int dhu4 = (ctas >= 96 && ctas <= 256) ? 1 : 0;
    if (const char* e = getenv("KDA_G2_R1A_DHU")) dhu4 = atoi(e);""", """    int dhu4 = (ctas >= 96 && ctas <= 256) ? 1 : 0;
    // kra P-67: chain-bound TP8 -> the 4-wave form with SG / KB
    const bool tp8 = ctas <= g2_env_long("KDA_G2_DHU4_TP8_MAX_CTAS", 80);
    if (tp8) dhu4 = 1;
    if (const char* e = getenv("KDA_G2_R1A_DHU")) dhu4 = atoi(e);""")
s = sub(s, """        if (lin) hipLaunchKernelGGL((g2_dhu_v8b4_kernel<0, true>), dim3(kK / kBV, nseg * H), dim3(256), dyn((g2_dhu_v8b4_kernel<0, true>)), s, p);""",
        """        if (tp8 && lin) hipLaunchKernelGGL((g2_dhu_v8b4_kernel<0, true, true, true>), dim3(kK / kBV, nseg * H), dim3(256), dyn((g2_dhu_v8b4_kernel<0, true, true, true>)), s, p);   // kra P-67
        else if (lin) hipLaunchKernelGGL((g2_dhu_v8b4_kernel<0, true>), dim3(kK / kBV, nseg * H), dim3(256), dyn((g2_dhu_v8b4_kernel<0, true>)), s, p);""")
f.write_text(s)
print("P-67 APPLIED")
