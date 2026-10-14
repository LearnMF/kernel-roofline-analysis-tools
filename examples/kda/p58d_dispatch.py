"""kra P-58 (shape-dispatched, bitwise): g2_fwd_h_v8o4_kernel gets a template flag SG -- wave index via readfirstlane so
the per-step addresses keep their uniform part in SGPRs.  Used for modes 1 / 4 only when 4 * nseg * H <= 80 (at most
one CTA per CU, chain-bound TP8 regime); with several CTAs per CU the extra SALU cost MODE 4 4-9% (P-58 probe).
    python3 p58d_dispatch.py <tree>"""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "csrc" / "g2" / "g2_fwd_v8.cuh"
s = p.read_text()
sig = ("template <bool VL, int MODE = 0, bool GE = true>   // kra P-52: GE = Gn holds exp2f(G)\n"
       "__global__ void __launch_bounds__(256) g2_fwd_h_v8o4_kernel(FwdOParams p) {")
assert s.count(sig) == 1
s = s.replace(sig, sig.replace("bool GE = true>", "bool GE = true, bool SG = false>   // kra P-58: SG = SGPR wave index"))
head, tail = s.split("g2_fwd_h_v8o4_kernel(FwdOParams p) {", 1)
old = "const int tid = threadIdx.x, wv = tid >> 6, l = tid & 63;"
assert old in tail.split("g2_fwd_h_v8o4w_kernel")[0]
tail = tail.replace(old, "const int tid = threadIdx.x, wv = SG ? __builtin_amdgcn_readfirstlane(tid >> 6) : (tid >> 6), "
                         "l = tid & 63;   // kra P-58", 1)
s = head + "g2_fwd_h_v8o4_kernel(FwdOParams p) {" + tail
reps = [
    ("    auto v8o4 = [&](auto vl) {\n        constexpr bool VL = decltype(vl)::value;",
     "    // kra P-58: SGPR wave index only in the chain-bound regime (at most one CTA per CU)\n"
     "    const bool sg = 4L * nseg * H <= g2_env_long(\"KDA_G2_SGPR_MAX_CTAS\", 80);\n"
     "    auto v8o4 = [&](auto vl) {\n        constexpr bool VL = decltype(vl)::value;"),
    ("            case 1: launch4(g2_fwd_h_v8o4_kernel<VL, 1>); break;",
     "            case 1: if (sg) launch4(g2_fwd_h_v8o4_kernel<VL, 1, true, true>);\n"
     "                    else    launch4(g2_fwd_h_v8o4_kernel<VL, 1>);\n                    break;"),
    ("                    else      launch4(g2_fwd_h_v8o4_kernel<VL, 4>);",
     "                    else if (sg) launch4(g2_fwd_h_v8o4_kernel<VL, 4, true, true>);\n"
     "                    else      launch4(g2_fwd_h_v8o4_kernel<VL, 4>);"),
]
for o, n in reps:
    assert s.count(o) == 1, o[:60]
    s = s.replace(o, n)
p.write_text(s)
print("P-58d APPLIED")
