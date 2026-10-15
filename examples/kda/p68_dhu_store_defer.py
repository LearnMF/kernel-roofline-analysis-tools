"""kra P-68 probe (bitwise, issue order only): the TP8 dhu (8-wave v8b, LIN instance) issues its HbN / aN(dv2)
global stores AFTER the prefetch loads that follow them.

SQTT (cpdhu_p65_d1): after barrier A the bN(dh) stores go out first and the KgA / dv / Gn prefetch loads behind them
wait up to ~400 cycles to issue; one store alone waits 601 cycles (46% of the 3753-cycle step is VMEM issue stall).
On gfx9 vmcnt also counts stores in issue order, so any later wait on those loads also waits for the stores (R52).
Here the bN values are read from dhB at the same place but stored after loadKA(nxt); the LIN aN(dv2) values are
gathered at the same place but stored after loadQW(nxt).  Values unchanged -> bit-identical.
Template flag DS on g2_dhu_v8b_kernel; used for the production LIN instance when 4*nseg*H <= 80.
    python3 p68_dhu_store_defer.py <tree>"""
import sys
from pathlib import Path

f = Path(sys.argv[1]) / "csrc" / "g2" / "g2_engine.cuh"
s = f.read_text()


def sub(b, old, new, cnt=1):
    assert b.count(old) == cnt, (old[:80], b.count(old))
    return b.replace(old, new)


s = sub(s, """__global__ void __launch_bounds__(512) g2_dhu_v8b_kernel(BwdNParams p) {""",
        """__global__ void __launch_bounds__(512) g2_dhu_v8b_kernel(BwdNParams p) {   // DS: kra P-68""")
# template line right above
i = s.index("__global__ void __launch_bounds__(512) g2_dhu_v8b_kernel(BwdNParams p) {")
j = s.rindex("template <", 0, i)
tmpl = s[j:i]
assert "LIN" in tmpl, tmpl
s = s[:j] + tmpl.rstrip()[:-1] + ", bool DS = false>\n" + s[i:]
head, rest = s.split("g2_dhu_v8b_kernel(BwdNParams p) {   // DS: kra P-68", 1)
body, tail = rest.split("// ================================================================ R1a-dhu: 4-wave form", 1)
loop_bn = """        lds_barrier();                                   // dhB (entry state of chunk i) + dOs ready
        // bN(dh) emit: slot (kt=wv, vt, l, e) = dh[16*wv + pi(m)][16*(2*vs+vt) + 4*(l>>4) + e]
        if (hasBN) {
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
        }"""
new_bn = """        lds_barrier();                                   // dhB (entry state of chunk i) + dOs ready
        // bN(dh) emit: slot (kt=wv, vt, l, e) = dh[16*wv + pi(m)][16*(2*vs+vt) + 4*(l>>4) + e]
        v4bh bnv[2];   // kra P-68: DS stores them after loadKA(nxt)
        if (hasBN) {
            const int q4_ = l >> 4, m_ = l & 15;
            #pragma unroll
            for (int vt = 0; vt < 2; ++vt) {
                v4bh val;
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    val[e] = reinterpret_cast<const short*>(dhB)[(16 * vt + 4 * q4_ + e) * kPadK + 16 * wv + pi16(m_)];
                const size_t off = ((((size_t)i * p.H + bh) * 8 + wv) * 8 + (2 * vs + vt)) * 256 + l * 4;
                if constexpr (DS) bnv[vt] = val;
                else *const_cast<v4bh*>(reinterpret_cast<const v4bh*>(p.HbN + off)) = val;
            }
        }"""
body = sub(body, loop_bn, new_bn)
body = sub(body, """        loadKA(nxt);
        {
            v4bh x;""", """        loadKA(nxt);
        if constexpr (DS) if (hasBN) {   // kra P-68: deferred bN(dh) stores
            #pragma unroll
            for (int vt = 0; vt < 2; ++vt) {
                const size_t off = ((((size_t)i * p.H + bh) * 8 + wv) * 8 + (2 * vs + vt)) * 256 + l * 4;
                *const_cast<v4bh*>(reinterpret_cast<const v4bh*>(p.HbN + off)) = bnv[vt];
            }
        }
        {
            v4bh x;""")
body = sub(body, """        if (hasAN2 && LIN) {   // kra P-22c/d: aN(dv2) as linear 8 B stores gathered from the staged vB tile
            #pragma unroll
            for (int v1g = v1; v1g < v1 + 1; ++v1g) {
                v4bh an;
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    an[e] = reinterpret_cast<const short*>(vB)[(16 * v1g + 4 * (l >> 4) + e) * kPadT + 16 * r1 + (l & 15)];
                *reinterpret_cast<v4bh*>(p.ANv2 + ((size_t)i * p.H + bh) * 8192 + (8 * r1 + 2 * vs + v1g) * 256 + l * 4) = an;
            }
        }""", """        v4bh anv;   // kra P-68: DS stores it after loadQW(nxt)
        if (hasAN2 && LIN) {   // kra P-22c/d: aN(dv2) as linear 8 B stores gathered from the staged vB tile
            #pragma unroll
            for (int v1g = v1; v1g < v1 + 1; ++v1g) {
                v4bh an;
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    an[e] = reinterpret_cast<const short*>(vB)[(16 * v1g + 4 * (l >> 4) + e) * kPadT + 16 * r1 + (l & 15)];
                if constexpr (DS) anv = an;
                else *reinterpret_cast<v4bh*>(p.ANv2 + ((size_t)i * p.H + bh) * 8192 + (8 * r1 + 2 * vs + v1g) * 256 + l * 4) = an;
            }
        }""")
body = sub(body, """        loadQW(nxt);
        write_dh();
    }
}""", """        loadQW(nxt);
        if constexpr (DS) if (hasAN2 && LIN)   // kra P-68: deferred aN(dv2) store
            *reinterpret_cast<v4bh*>(p.ANv2 + ((size_t)i * p.H + bh) * 8192 + (8 * r1 + 2 * vs + v1) * 256 + l * 4) = anv;
        write_dh();
    }
}""")
s = head + "g2_dhu_v8b_kernel(BwdNParams p) {   // DS: kra P-68" + body + "// ================================================================ R1a-dhu: 4-wave form" + tail
s = sub(s, """        if (prod && lin) hipLaunchKernelGGL((g2_dhu_v8b_kernel<1, true>), dim3(kK / kBV, nseg * H), dim3(512), dyn((g2_dhu_v8b_kernel<1, true>)), s, p);""",
        """        if (prod && lin && ctas <= g2_env_long("KDA_G2_DHU_DS_MAX_CTAS", 80))   // kra P-68
            hipLaunchKernelGGL((g2_dhu_v8b_kernel<1, true, true>), dim3(kK / kBV, nseg * H), dim3(512), dyn((g2_dhu_v8b_kernel<1, true, true>)), s, p);
        else if (prod && lin) hipLaunchKernelGGL((g2_dhu_v8b_kernel<1, true>), dim3(kK / kBV, nseg * H), dim3(512), dyn((g2_dhu_v8b_kernel<1, true>)), s, p);""")
f.write_text(s)
print("P-68 APPLIED")
