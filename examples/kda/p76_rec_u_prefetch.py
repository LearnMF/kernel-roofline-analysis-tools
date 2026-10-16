"""kra P-76 probe (bitwise, load placement only): the TP8 recompute recurrence's U prefetch.

A2 (results/kda-gfx936-20261016-a2-hscale) + vm_wait_map.py: the recompute step's largest critical item (306 cycles/
step) is the s_waitcnt vmcnt(14) right after the next step's W loads, waiting for loadU(nxt) -- issued ~2000 cycles
earlier, right AFTER this step's two Vn stores (vmcnt counts loads and stores together, retired in order: R52).  Template
int UV on g2_fwd_h_v8o4_kernel, used by the production TP8 recompute instance (MODE 4, SG, KP, BP) and selected by
KDA_G2_REC_UV (probe):
  1  U two steps ahead: un0/un1 hold chunk i+1 (loaded during step i-1), uf <- un after v_new, then un <- U(i+2)
  2  loadU(nxt) issued before the Vn stores (right after x0/x1 are formed)
    python3 p76_rec_u_prefetch.py <tree>"""
import sys
from pathlib import Path

f = Path(sys.argv[1]) / "csrc" / "g2" / "g2_fwd_v8.cuh"
s = f.read_text()


def sub(b, old, new, cnt=1):
    assert b.count(old) == cnt, (old[:80], b.count(old))
    return b.replace(old, new)


s = sub(s, """template <bool VL, int MODE = 0, bool GE = true, bool SG = false, bool PF = false, bool KP = false, bool VP = false, bool BP = false>   // kra P-58: SG = SGPR wave index   // kra P-52: GE = Gn holds exp2f(G)
__global__ void __launch_bounds__(256) g2_fwd_h_v8o4_kernel(FwdOParams p) {""",
        """template <bool VL, int MODE = 0, bool GE = true, bool SG = false, bool PF = false, bool KP = false, bool VP = false, bool BP = false, int UV = 0>   // kra P-58: SG = SGPR wave index   // kra P-52: GE = Gn holds exp2f(G)   // kra P-76: UV
__global__ void __launch_bounds__(256) g2_fwd_h_v8o4_kernel(FwdOParams p) {""")
head, rest = s.split("__global__ void __launch_bounds__(256) g2_fwd_h_v8o4_kernel(FwdOParams p) {", 1)
body, tail = rest.split("// kra P-49: g2_fwd_h_v8o4_kernel with NVH", 1)
body = sub(body, """    loadW(c0); loadU(c0, 0, uf0); loadU(c0, 1, uf1);
""", """    loadW(c0); loadU(c0, 0, uf0); loadU(c0, 1, uf1);
    v4bh un0, un1;   // kra P-76 UV=1: U of chunk i+1
    auto ldU2 = [&](int i_, v4bh& a0, v4bh& a1) {
        ld16x2(p.Un + (((chunk + i_) * 4 + r1) * 8 + 2 * vs) * 256 + l * 8, a0, a1);
    };
    if constexpr (UV == 1) ldU2(c0 + 1 < c1 ? c0 + 1 : c0, un0, un1);
""")
body = sub(body, """                reinterpret_cast<short*>(vB)[(16 * 1 + vc) * kPadT + 16 * r1 + m] = x1[e];
            }
            if constexpr (!VP) {""", """                reinterpret_cast<short*>(vB)[(16 * 1 + vc) * kPadT + 16 * r1 + m] = x1[e];
            }
            if constexpr (UV == 2) { loadU(nxt, 0, uf0); loadU(nxt, 1, uf1); }   // kra P-76: ahead of the Vn stores
            if constexpr (!VP) {""")
body = sub(body, """        if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);   // kra P-11: keep load spacing
        loadU(nxt, 0, uf0); loadU(nxt, 1, uf1);
""", """        if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);   // kra P-11: keep load spacing
        if constexpr (UV == 1) {   // kra P-76: U two steps ahead
            uf0 = un0; uf1 = un1;
            ldU2(i + 2 < c1 ? i + 2 : c1 - 1, un0, un1);
        } else if constexpr (UV == 0) { loadU(nxt, 0, uf0); loadU(nxt, 1, uf1); }
""")
s = head + "__global__ void __launch_bounds__(256) g2_fwd_h_v8o4_kernel(FwdOParams p) {" + body + "// kra P-49: g2_fwd_h_v8o4_kernel with NVH" + tail
s = sub(s, """                    else if (sg) { if (kp) launch4(g2_fwd_h_v8o4_kernel<VL, 4, true, true, false, true, false, true>);""",
        """                    else if (sg) { const long uv = g2_env_long("KDA_G2_REC_UV", 0);   // kra P-76 probe
                                   if (kp && uv == 1) launch4(g2_fwd_h_v8o4_kernel<VL, 4, true, true, false, true, false, true, 1>);
                                   else if (kp && uv == 2) launch4(g2_fwd_h_v8o4_kernel<VL, 4, true, true, false, true, false, true, 2>);
                                   else if (kp) launch4(g2_fwd_h_v8o4_kernel<VL, 4, true, true, false, true, false, true>);""")
f.write_text(s)
print("P-76 APPLIED")
