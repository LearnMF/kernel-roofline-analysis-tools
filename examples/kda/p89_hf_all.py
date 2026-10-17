"""kra P-89 probe (bitwise): P-88's head-fastest grid for the other chunk-parallel kernels -- wu, dav, bwd_tail,
fwd prep (a and b share one grid), head gate -- each behind its own switch (KDA_G2_HF_WU / _DAV / _TAIL / _PREP /
_GATE, default 0).  Every (chunk, head) CTA of these kernels is independent; the cross-CTA reductions (tail's
dA_part / ddt_part, prep / gate none) are reduced afterwards in fixed order, so values are unchanged.
    python3 p89_hf_all.py <dir holding the g2 sources>"""
import sys
from pathlib import Path

d = Path(sys.argv[1])


def edit(name, pairs):
    f = d / name
    s = f.read_text()
    for old, new, cnt in pairs:
        assert s.count(old) == cnt, (name, old[:80], s.count(old))
        s = s.replace(old, new)
    f.write_text(s)


A = "    const int i = blockIdx.x, hh = blockIdx.y, tid = threadIdx.x;"
A_NEW = ("    const bool hf_ = gridDim.x == (unsigned)p.H && gridDim.y != (unsigned)p.H;   // kra P-89: head-fastest grid\n"
         "    const int i = hf_ ? (int)blockIdx.y : (int)blockIdx.x, hh = hf_ ? (int)blockIdx.x : (int)blockIdx.y, tid = threadIdx.x;")
B = "    const int i = blockIdx.x, hh = blockIdx.y;"
B_NEW = ("    const bool hf_ = gridDim.x == (unsigned)p.H && gridDim.y != (unsigned)p.H;   // kra P-89: head-fastest grid\n"
         "    const int i = hf_ ? (int)blockIdx.y : (int)blockIdx.x, hh = hf_ ? (int)blockIdx.x : (int)blockIdx.y;")
edit("g2_wu.cuh", [(A, A_NEW, 1),
                   ("    const bool ks = g2::g2_env_long(\"KDA_G2_WU_KS\", 1)",
                    "    const dim3 wgrd = g2::g2_env_long(\"KDA_G2_HF_WU\", 0) ? dim3(H, T / 64) : dim3(T / 64, H);   // kra P-89\n"
                    "    const bool ks = g2::g2_env_long(\"KDA_G2_WU_KS\", 1)", 1),
                   ("dim3(T / 64, H), dim3(256), 0, s, p);", "wgrd, dim3(256), 0, s, p);", 6)])
edit("g2_dav.cuh", [(A, A_NEW, 1),
                    ("    if (ctok)\n        hipLaunchKernelGGL(g2::g2_dav_kernel<true>, dim3(T / 64, H),",
                     "    const dim3 dgrd = g2::g2_env_long(\"KDA_G2_HF_DAV\", 0) ? dim3(H, T / 64) : dim3(T / 64, H);   // kra P-89\n"
                     "    if (ctok)\n        hipLaunchKernelGGL(g2::g2_dav_kernel<true>, dgrd,", 1),
                    ("hipLaunchKernelGGL(g2::g2_dav_kernel<false>, dim3(T / 64, H),", "hipLaunchKernelGGL(g2::g2_dav_kernel<false>, dgrd,", 1)])
edit("g2_bwd_tail.cuh", [(A, A_NEW, 1),
                         ("    if (ctok)\n        hipLaunchKernelGGL(g2::g2_bwd_tail_kernel<true>, dim3(NT, H),",
                          "    const dim3 tgrd = g2::g2_env_long(\"KDA_G2_HF_TAIL\", 0) ? dim3(H, NT) : dim3(NT, H);   // kra P-89\n"
                          "    if (ctok)\n        hipLaunchKernelGGL(g2::g2_bwd_tail_kernel<true>, tgrd,", 1),
                         ("hipLaunchKernelGGL(g2::g2_bwd_tail_kernel<false>, dim3(NT, H),",
                          "hipLaunchKernelGGL(g2::g2_bwd_tail_kernel<false>, tgrd,", 1)])
edit("g2_fwd_prep.cuh", [(B, B_NEW, 2),
                         ("    const dim3 grid(NT, H), blk(256);",
                          "    const dim3 grid = g2::g2_env_long(\"KDA_G2_HF_PREP\", 0) ? dim3(H, NT) : dim3(NT, H), blk(256);   // kra P-89", 1)])
edit("g2_head.cuh", [(B, B_NEW, 1),
                     ("    const dim3 grid(NT, H), blk(256);",
                      "    const dim3 grid = g2::g2_env_long(\"KDA_G2_HF_GATE\", 0) ? dim3(H, NT) : dim3(NT, H), blk(256);   // kra P-89", 1)])
print("P-89 APPLIED")
