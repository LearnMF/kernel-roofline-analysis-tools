"""C-3 probe: forward MODE 1 through v8o4w with NVH = 1 (16-column V-slabs, 8 CTAs/head) or 2 (32 columns,
4 CTAs/head) forced by KDA_G2_C3_NVH -- isolates the effect of the CTA count (96 vs 48 at H = 12)."""
import sys
from pathlib import Path
f = Path(sys.argv[1]) / "csrc" / "g2" / "g2_fwd_v8.cuh"
s = f.read_text()
old = """        #pragma unroll
        for (int pr = 0; pr < NVH / 2; ++pr)
            ld16x2(p.Un + (((chunk + i) * 4 + r1) * 8 + NVH * vs + 2 * pr) * 256 + l * 8, uf[2 * pr], uf[2 * pr + 1]);"""
assert s.count(old) == 1
s = s.replace(old, """        if constexpr (NVH == 1) {   // C-3: one V-tile of the pair (paired nat16 layout)
            uf[0] = ld8(p.Un + (((chunk + i) * 4 + r1) * 8 + (vs & ~1)) * 256 + l * 8 + 4 * (vs & 1));
        } else {
        #pragma unroll
        for (int pr = 0; pr < NVH / 2; ++pr)
            ld16x2(p.Un + (((chunk + i) * 4 + r1) * 8 + NVH * vs + 2 * pr) * 256 + l * 8, uf[2 * pr], uf[2 * pr + 1]);
        }""")
old2 = "    if (r1a && !kp && H >= g2_env_long(\"KDA_G2_V8O4W_MIN_H\", 40)"
assert s.count(old2) == 1
s = s.replace(old2, """    if (mode == 1 && g2_env_long("KDA_G2_C3_NVH", 0) > 0) {   // C-3 probe
        const long nvh = g2_env_long("KDA_G2_C3_NVH", 0);
        const dim3 g3(kK / (16 * (unsigned)nvh), (unsigned)nseg * H);
        if (ctok != nullptr) {
            if (nvh == 1) hipLaunchKernelGGL((g2_fwd_h_v8o4w_kernel<true, 1, 1>), g3, dim3(256), 0, s, p);
            else          hipLaunchKernelGGL((g2_fwd_h_v8o4w_kernel<true, 1, 2>), g3, dim3(256), 0, s, p);
        } else {
            if (nvh == 1) hipLaunchKernelGGL((g2_fwd_h_v8o4w_kernel<false, 1, 1>), g3, dim3(256), 0, s, p);
            else          hipLaunchKernelGGL((g2_fwd_h_v8o4w_kernel<false, 1, 2>), g3, dim3(256), 0, s, p);
        }
        return (int)hipGetLastError();
    }
""" + old2)
f.write_text(s)
print("C3 APPLIED")
