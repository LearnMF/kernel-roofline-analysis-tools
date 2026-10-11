"""kra P-24 (round 7, structural, BW-L2-specific): cap the per-CU concurrency of the per-head
recurrences at deep saturation so that the K-side operands shared by a head's 4 V-slab CTAs stay
in the 8 MB L2 (measured: 1.00x interface bytes at 8K/12, 1.06-1.11x at H=48, 1.33-1.42x at H=64,
1.35-1.85x at H=96).  Host-side only: a dynamic-LDS reservation (no kernel uses dynamic LDS)
lifts each CTA to 24 KB total (v8o4: <= 2 CTAs/CU) / 33 KB (dhu: <= 1 CTA/CU); the static LDS is
read with hipFuncGetAttributes.  Thresholds (CTAs = 4*nseg*H): v8o4 >= 224, dhu >= 192;
KDA_G2_V8O4_THROTTLE_MIN_CTAS / KDA_G2_DHU_THROTTLE_MIN_CTAS override (0 = always, huge = never).
Kernels untouched -> bit-identical.        Usage: python3 p24_l2_throttle.py <tree root>"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
V8 = ROOT / "csrc/g2/g2_fwd_v8.cuh"
EN = ROOT / "csrc/g2/g2_engine.cuh"
CM = ROOT / "csrc/g2/g2_common.h"


def sub(path, old, new, count=1):
    t = path.read_text()
    assert t.count(old) == count, (path.name, old[:80], t.count(old))
    path.write_text(t.replace(old, new))


# shared helper: dynamic LDS that tops a kernel up to `target_total` bytes per CTA
helper = """
// kra P-24: L2-aware concurrency cap.  Returns the dynamic-LDS bytes that lift a kernel's static
// LDS to `target_total` per CTA (so that floor(64 KB / target_total) CTAs fit per CU); 0 if the
// kernel is already larger.  The static size comes from the code object (cached per kernel).
template <typename K>
inline size_t g2_lds_topup(K kernel, size_t target_total) {
    hipFuncAttributes a{};
    if (hipFuncGetAttributes(&a, reinterpret_cast<const void*>(kernel)) != hipSuccess) return 0;
    return a.sharedSizeBytes < target_total ? target_total - a.sharedSizeBytes : 0;
}
inline long g2_env_long(const char* name, long dflt) {
    const char* e = getenv(name);
    return e ? atol(e) : dflt;
}
"""
t = CM.read_text()
anchor = "__device__ __forceinline__ size_t g2_nat_blk(int i, int hh, int H, int NT,"
assert t.count(anchor) == 1
CM.write_text(t.replace(anchor, helper + "\n" + anchor))

# v8o4 launches
sub(V8, """    auto v8o4 = [&](auto vl) {
        constexpr bool VL = decltype(vl)::value;
        switch (mode) {
            case 1: hipLaunchKernelGGL((g2_fwd_h_v8o4_kernel<VL, 1>), grid, dim3(256), 0, s, p); break;
            case 2: hipLaunchKernelGGL((g2_fwd_h_v8o4_kernel<VL, 2>), grid, dim3(256), 0, s, p); break;
            case 3: hipLaunchKernelGGL((g2_fwd_h_v8o4_kernel<VL, 3>), grid, dim3(256), 0, s, p); break;
            case 4: hipLaunchKernelGGL((g2_fwd_h_v8o4_kernel<VL, 4>), grid, dim3(256), 0, s, p); break;
            default: hipLaunchKernelGGL((g2_fwd_h_v8o4_kernel<VL, 0>), grid, dim3(256), 0, s, p); break;
        }
    };""",
    """    // kra P-24: at deep saturation cap v8o4 at 2 CTAs/CU (24 KB LDS per CTA) so the K-side
    // operands shared by a head's 4 V-slab CTAs stay in L2.
    const bool throttle = 4L * nseg * H >= g2_env_long("KDA_G2_V8O4_THROTTLE_MIN_CTAS", 224);
    auto launch4 = [&](auto kern) {
        const size_t dyn = throttle ? g2_lds_topup(kern, 24 * 1024) : 0;
        hipLaunchKernelGGL(kern, grid, dim3(256), dyn, s, p);
    };
    auto v8o4 = [&](auto vl) {
        constexpr bool VL = decltype(vl)::value;
        switch (mode) {
            case 1: launch4(g2_fwd_h_v8o4_kernel<VL, 1>); break;
            case 2: launch4(g2_fwd_h_v8o4_kernel<VL, 2>); break;
            case 3: launch4(g2_fwd_h_v8o4_kernel<VL, 3>); break;
            case 4: launch4(g2_fwd_h_v8o4_kernel<VL, 4>); break;
            default: launch4(g2_fwd_h_v8o4_kernel<VL, 0>); break;
        }
    };""")

# dhu launches
t = EN.read_text()
i = t.index('extern "C" int g2_dhu_v8b(')
head, tail = t[:i], t[i:]
old = """    // kra P-22d: linear aN(dv2) emit only in the latency-bound regime"""
assert tail.count(old) == 1
tail = tail.replace(old, """    // kra P-24: at deep saturation cap dhu at 1 CTA/CU (33 KB LDS per CTA): the operands shared
    // by a head's 4 V-slab CTAs then stay in L2.
    const bool dthrottle = ctas >= g2_env_long("KDA_G2_DHU_THROTTLE_MIN_CTAS", 192);
    auto dyn = [&](auto kern) -> size_t { return dthrottle ? g2_lds_topup(kern, 33 * 1024) : 0; };
""" + old)
import re
n = 0
def repl(mo):
    global n
    n += 1
    kern, blk = mo.group(1), mo.group(2)
    return f"hipLaunchKernelGGL({kern}, dim3(kK / kBV, nseg * H), dim3({blk}), dyn({kern}), s, p);"
tail = re.sub(r"hipLaunchKernelGGL\((\(?g2_dhu_v8b4?_kernel<[^>]*>\)?), dim3\(kK / kBV, nseg \* H\), dim3\((\d+)\), 0, s, p\);", repl, tail)
assert n == 6, n
EN.write_text(head + tail)
print("P-24 APPLIED", n)
