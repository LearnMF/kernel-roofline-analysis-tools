"""kra P-80a ABLATION (timing only; KDA_G2_DHU_WBT gives WRONG dhu numbers): price option b -- wu stops writing Wb (the
B form of W, read only by dhu) and dhu forms its B fragments from Wn with an in-wave LDS transpose.
  KDA_G2_WU_NOWB=1   wu skips the Wb emit (its LDS transpose round trip + 2 barriers + 8 KB/CTA store)
  KDA_G2_DHU_WBT=1   TP8 dhu (v8b LIN) runs the transpose dhu would need, on its wb fragments, mid-step (after GEMM-A):
                     per lane 16 ds_write_b16 into a [k][t] tile (as an A-form Wn source would), 4 ds_read_b64 back
    python3 p80a_wb_ablate.py <tree>"""
import sys
from pathlib import Path

G = Path(sys.argv[1]) / "csrc" / "g2"


def sub(path, old, new, cnt=1):
    s = path.read_text()
    assert s.count(old) == cnt, (path.name, old[:90], s.count(old))
    path.write_text(s.replace(old, new))


wu = G / "g2_wu.cuh"
sub(wu, """    int kp = 0;    // kra P-61e: Kn lane-permuted (H <= KPERM_MAX_H)
};""", """    int kp = 0;    // kra P-61e: Kn lane-permuted (H <= KPERM_MAX_H)
    int nowb = 0;  // kra P-80a ablation
};""")
sub(wu, "g2_glb(p_.ctok), g2_glb(p_.clen), p_.stop, p_.kp};",
    "g2_glb(p_.ctok), g2_glb(p_.clen), p_.stop, p_.kp, p_.nowb};   // kra P-80a")
sub(wu, "        mmac_phase(p.Wn, p.Wb, nullptr, 2);", "        mmac_phase(p.Wn, p.nowb ? nullptr : p.Wb, nullptr, 2);   // kra P-80a")
sub(wu, """    p.kp = H <= g2::g2_env_long("KDA_G2_KPERM_MAX_H", 20);   // kra P-61e
""", """    p.kp = H <= g2::g2_env_long("KDA_G2_KPERM_MAX_H", 20);   // kra P-61e
    p.nowb = (int)g2::g2_env_long("KDA_G2_WU_NOWB", 0);       // kra P-80a ablation
""")

en = G / "g2_engine.cuh"
s = en.read_text()
s = s.replace("""template <int MODE = 0, bool LIN = false>
__global__ void __launch_bounds__(512) g2_dhu_v8b_kernel(BwdNParams p) {""", """template <int MODE = 0, bool LIN = false, bool WBT = false>   // kra P-80a: WBT ablation
__global__ void __launch_bounds__(512) g2_dhu_v8b_kernel(BwdNParams p) {""", 1)
head, rest = s.split("__global__ void __launch_bounds__(512) g2_dhu_v8b_kernel(BwdNParams p) {", 1)
body, tail = rest.split("// ================================================================ R1a-dhu: 4-wave form", 1)
o = """    __shared__ __align__(16) v4bh dOs[2][8 * 64];        // dO^T A-frags [vt*4+rt][lane], double buffered
"""
assert body.count(o) == 1
body = body.replace(o, o + """    __shared__ __align__(16) short wbT[WBT ? 8 : 1][WBT ? 4 : 1][16 * 20];   // kra P-80a: per-wave [k][t] tiles
""")
o = """        loadKA(nxt);
        {
            v4bh x;"""
assert body.count(o) == 1
body = body.replace(o, """        loadKA(nxt);
        if constexpr (WBT) {   // kra P-80a: the in-wave transpose an A-form (Wn) source would need
            const int m_ = l & 15, q_ = l >> 4;
            #pragma unroll
            for (int rt = 0; rt < 4; ++rt)
                #pragma unroll
                for (int e = 0; e < 4; ++e) wbT[wv][rt][(4 * q_ + e) * 20 + m_] = wb[rt][e];
            #pragma unroll
            for (int rt = 0; rt < 4; ++rt) wb[rt] = *reinterpret_cast<const v4bh*>(&wbT[wv][rt][m_ * 20 + 4 * q_]);
        }
        {
            v4bh x;""")
s = head + "__global__ void __launch_bounds__(512) g2_dhu_v8b_kernel(BwdNParams p) {" + body + \
    "// ================================================================ R1a-dhu: 4-wave form" + tail
o = """        if (prod && lin) hipLaunchKernelGGL((g2_dhu_v8b_kernel<1, true>), dim3(kK / kBV, nseg * H), dim3(512), dyn((g2_dhu_v8b_kernel<1, true>)), s, p);"""
assert s.count(o) == 1
s = s.replace(o, """        if (prod && lin && g2_env_long("KDA_G2_DHU_WBT", 0)) hipLaunchKernelGGL((g2_dhu_v8b_kernel<1, true, true>), dim3(kK / kBV, nseg * H), dim3(512), dyn((g2_dhu_v8b_kernel<1, true, true>)), s, p);   // kra P-80a
        else """ + o.lstrip())
en.write_text(s)
print("P-80a APPLIED")
