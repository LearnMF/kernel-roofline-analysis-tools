"""kra P-84c probe (bitwise): wu phase 3 without Kn (saved-h / keep-state backward) uses the carried exp2f(g) and
issues its q quads first (template bool QS, KDA_G2_WU_QS=1).

SQTT round 2 (results/kda-ceiling-cards-20261016-p82 §4, map_waits -> g2_wu_hip.cuh:368/407): wu still waits on VMEM
46% of its wave-cycles, 18.8% at one site 8x per wave -- phase 3's `else` branch (p.Kn == nullptr).  Since P-82 the
saved-h route makes that branch the default at H <= 24: it calls qg_of() per quad in a rolled loop, re-reading q AND
the fp32 g (101 MB at 16K/12) and paying one full load latency per quad.  The Kn branch already uses phase 1's
carried exp2f(g) (P-19, gated bit-identical to qg_of).  QS: all 8 q quads in flight first, then
qg = bf_rne(bf2f(q) * egr[it]) into the tile -- the Kn branch's expression; plus the barrier the Kn branch has
before overwriting the slab (the u-phase MMAC reads it; without VNEW nothing else orders them).
    python3 p84c_wu_qs.py <dir holding g2_wu.cuh>"""
import sys
from pathlib import Path

f = Path(sys.argv[1]) / "g2_wu.cuh"
s = f.read_text()


def sub(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (old[:90], s.count(old))
    s = s.replace(old, new)


tmpl_old = "template <bool VL, bool KS = false>   // kra P-84b: KS"
if tmpl_old in s:
    sub(tmpl_old, "template <bool VL, bool KS = false, bool QS = false>   // kra P-84b: KS; P-84c: QS")
    launch_old = """    if (ctok) {
        if (ks) hipLaunchKernelGGL((g2::g2_wu_kernel<true, true>), dim3(T / 64, H), dim3(256), 0, s, p);
        else    hipLaunchKernelGGL(g2::g2_wu_kernel<true>, dim3(T / 64, H), dim3(256), 0, s, p);
    } else {
        if (ks) hipLaunchKernelGGL((g2::g2_wu_kernel<false, true>), dim3(T / 64, H), dim3(256), 0, s, p);
        else    hipLaunchKernelGGL(g2::g2_wu_kernel<false>, dim3(T / 64, H), dim3(256), 0, s, p);
    }"""
    launch_new = """    const bool qs = g2::g2_env_long("KDA_G2_WU_QS", 0) != 0;   // kra P-84c probe
    if (ctok) {
        if (ks && qs) hipLaunchKernelGGL((g2::g2_wu_kernel<true, true, true>), dim3(T / 64, H), dim3(256), 0, s, p);
        else if (ks)  hipLaunchKernelGGL((g2::g2_wu_kernel<true, true>), dim3(T / 64, H), dim3(256), 0, s, p);
        else          hipLaunchKernelGGL(g2::g2_wu_kernel<true>, dim3(T / 64, H), dim3(256), 0, s, p);
    } else {
        if (ks && qs) hipLaunchKernelGGL((g2::g2_wu_kernel<false, true, true>), dim3(T / 64, H), dim3(256), 0, s, p);
        else if (ks)  hipLaunchKernelGGL((g2::g2_wu_kernel<false, true>), dim3(T / 64, H), dim3(256), 0, s, p);
        else          hipLaunchKernelGGL(g2::g2_wu_kernel<false>, dim3(T / 64, H), dim3(256), 0, s, p);
    }"""
    sub(launch_old, launch_new)
else:
    raise SystemExit("needs P-84b (KS) first")
sub("""            } else {
                for (int t = tid; t < 2048; t += 256) {
                    short qgv[4];
                    qg_of(t, qgv);
                    #pragma unroll
                    for (int dv = 0; dv < 4; ++dv) tile[wu_tile_idx(ibn_of(t) + 4 * dv)] = qgv[dv];
                }
            }""", """            } else if constexpr (QS) {   // kra P-84c: carried exp2f(g) (the Kn branch's expression), q quads first
                v4bh qvs[8];
                #pragma unroll
                for (int it = 0; it < 8; ++it) {
                    const int tt = tid + 256 * it;
                    const int j = tt >> 5, vq = (tt & 31) << 2;
                    qvs[it] = g2_row_ok<M>(j, cl)
                        ? *reinterpret_cast<const v4bh*>(p.q + sbase + (size_t)j * rstride + vq) : v4bh{0, 0, 0, 0};
                }
                __syncthreads();                            // every wave's u-MMAC slab reads done
                #pragma unroll
                for (int it = 0; it < 8; ++it)
                    #pragma unroll
                    for (int dv = 0; dv < 4; ++dv)
                        tile[wu_tile_idx(ibn_of(tid + 256 * it) + 4 * dv)] = bf_rne(bf2f(qvs[it][dv]) * egr[it][dv]);
            } else {
                for (int t = tid; t < 2048; t += 256) {
                    short qgv[4];
                    qg_of(t, qgv);
                    #pragma unroll
                    for (int dv = 0; dv < 4; ++dv) tile[wu_tile_idx(ibn_of(t) + 4 * dv)] = qgv[dv];
                }
            }""")
f.write_text(s)
print("P-84c APPLIED")
