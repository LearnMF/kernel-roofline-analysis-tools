"""kra P-74 probe (bitwise): the TP8 dhu (8-wave v8b, production LIN instance) loads each KgA tile once per CTA.

dhu at TP8 16K/12 is VMEM-issue bound per CU (46% of its 3753-cycle step, cpdhu_p65_d1) and TA occupancy follows the
bytes moved (R54).  In v8b the waves (r1, v1 = 0) and (r1, v1 = 1) load the SAME KgA rows (4 x 16 B pairs per lane
per step = 32 KB of the CTA's ~80 KB of loads per step).  With KS each of the two waves loads one half (kt 0-3 /
kt 4-7) and publishes it through LDS (kaS[r1][kt][lane], 16 KB): written at the end of the step that prefetched it
(and after the prologue load), read back right after barrier A of the step that uses it; barrier B of that step
orders the reads before the next step's writes.  GEMM-A consumes ka[0..7] in the same order -> bit-identical.
Template flag KS on g2_dhu_v8b_kernel; used for the production LIN instance when 4*nseg*H <= KDA_G2_DHU_KS_MAX_CTAS
(80).      python3 p74_dhu_ka_share.py <tree>"""
import sys
from pathlib import Path

f = Path(sys.argv[1]) / "csrc" / "g2" / "g2_engine.cuh"
s = f.read_text()


def sub(b, old, new, cnt=1):
    assert b.count(old) == cnt, (old[:80], b.count(old))
    return b.replace(old, new)


s = sub(s, """template <int MODE = 0, bool LIN = false>
__global__ void __launch_bounds__(512) g2_dhu_v8b_kernel(BwdNParams p) {""",
        """template <int MODE = 0, bool LIN = false, bool KS = false>   // kra P-74: KS
__global__ void __launch_bounds__(512) g2_dhu_v8b_kernel(BwdNParams p) {""")
head, rest = s.split("__global__ void __launch_bounds__(512) g2_dhu_v8b_kernel(BwdNParams p) {", 1)
body, tail = rest.split("// ================================================================ R1a-dhu: 4-wave form", 1)
body = sub(body, """    __shared__ __align__(16) v4bh dOs[2][8 * 64];        // dO^T A-frags [vt*4+rt][lane], double buffered
""", """    __shared__ __align__(16) v4bh dOs[2][8 * 64];        // dO^T A-frags [vt*4+rt][lane], double buffered
    __shared__ __align__(16) v4bh kaS[KS ? 4 : 1][KS ? 8 : 1][64];   // kra P-74: KgA tiles, one load per CTA
""")
body = sub(body, """    auto loadKA = [&](int i) {
        const bf16* src = p.KgA + (((chunk + i) * 4 + r1) * 8) * 256 + l * 8;      // kra L16: paired
        #pragma unroll
        for (int j = 0; j < 4; ++j) ld16x2(src + j * 512, ka[2 * j], ka[2 * j + 1]);
    };""", """    // kra P-74: KS keeps this wave's half (v1 = 0: kt 0-3, v1 = 1: kt 4-7) in kown and the partner's in kpar;
    // static register indices only (a runtime-indexed ka[] goes to scratch), the per-kt select sits at the use.
    v4bh kown[4], kpar[4];
    auto loadKA = [&](int i) {
        const bf16* src = p.KgA + (((chunk + i) * 4 + r1) * 8) * 256 + l * 8;      // kra L16: paired
        if constexpr (KS) {
            ld16x2(src + (2 * v1) * 512, kown[0], kown[1]);
            ld16x2(src + (2 * v1 + 1) * 512, kown[2], kown[3]);
        } else {
        #pragma unroll
        for (int j = 0; j < 4; ++j) ld16x2(src + j * 512, ka[2 * j], ka[2 * j + 1]);
        }
    };
    auto ka_put = [&]() {   // kra P-74: publish this wave's half
        if constexpr (KS) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) kaS[r1][4 * v1 + j][l] = kown[j];
        }
    };
    auto ka_get = [&]() {   // kra P-74: the partner wave's half
        if constexpr (KS) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) kpar[j] = kaS[r1][4 * (1 - v1) + j][l];
        }
    };
    auto kaop = [&](int kt) -> v4bh {   // GEMM-A operand kt
        if constexpr (KS) return ((kt < 4) == (v1 == 0)) ? kown[kt & 3] : kpar[kt & 3];
        else return ka[kt];
    };""")
body = sub(body, """    loadKA(last); loadDV(last); loadG(last); loadQW(last); loadDO(last);
    write_dh();
""", """    loadKA(last); loadDV(last); loadG(last); loadQW(last); loadDO(last);
    write_dh();
    ka_put();   // kra P-74
""")
body = sub(body, """        loadDO(nxt);                                     // a full step ahead of its publish
        lds_barrier();                                   // dhB (entry state of chunk i) + dOs ready
""", """        loadDO(nxt);                                     // a full step ahead of its publish
        lds_barrier();                                   // dhB (entry state of chunk i) + dOs ready
        ka_get();   // kra P-74
""")
body = sub(body, """        loadQW(nxt);
        write_dh();
    }
}""", """        loadQW(nxt);
        write_dh();
        ka_put();   // kra P-74: chunk nxt's half (barrier B above ordered this step's reads)
    }
}""")
body = sub(body, """            acc = mmac(ka[kt], ld8(dhB + (16 * v1 + (l & 15)) * kPadK + 16 * kt + (l >> 4) * 4), acc);""",
           """            acc = mmac(kaop(kt), ld8(dhB + (16 * v1 + (l & 15)) * kPadK + 16 * kt + (l >> 4) * 4), acc);   // kra P-74""")
s = head + "__global__ void __launch_bounds__(512) g2_dhu_v8b_kernel(BwdNParams p) {" + body + \
    "// ================================================================ R1a-dhu: 4-wave form" + tail
s = sub(s, """        if (prod && lin) hipLaunchKernelGGL((g2_dhu_v8b_kernel<1, true>), dim3(kK / kBV, nseg * H), dim3(512), dyn((g2_dhu_v8b_kernel<1, true>)), s, p);""",
        """        if (prod && lin && ctas <= g2_env_long("KDA_G2_DHU_KS_MAX_CTAS", 80))   // kra P-74
            hipLaunchKernelGGL((g2_dhu_v8b_kernel<1, true, true>), dim3(kK / kBV, nseg * H), dim3(512), dyn((g2_dhu_v8b_kernel<1, true, true>)), s, p);
        else if (prod && lin) hipLaunchKernelGGL((g2_dhu_v8b_kernel<1, true>), dim3(kK / kBV, nseg * H), dim3(512), dyn((g2_dhu_v8b_kernel<1, true>)), s, p);""")
f.write_text(s)
print("P-74 APPLIED")
