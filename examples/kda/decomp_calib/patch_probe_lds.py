#!/usr/bin/env python3
"""P1 patch: KDA_PROBE_LDS_<name>=KB env switches that raise a kernel's per-CTA
LDS to KB bytes via g2_lds_topup, lowering its resident CTAs/CU.
Applied to a COPY tree only (e.g. /opt/kda_env/g2_calib).

    python3 patch_probe_lds.py /opt/kda_env/g2_calib
"""
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
G2 = ROOT / "csrc" / "g2"

HELPER = """
// kra P1-probe: env-overridable LDS topup. KDA_PROBE_LDS_<name>=KB (total LDS/CTA,
// kilobytes); returns the dynamic LDS to request, or `fallback` when unset/0.
template <typename K>
inline size_t g2_probe_lds(K kern, const char* env, size_t fallback) {
    const char* v = getenv(env);
    return (v && atol(v) > 0) ? g2_lds_topup(kern, (size_t)atol(v) * 1024) : fallback;
}
"""


def add_helper():
    f = G2 / "g2_common.h"
    s = f.read_text()
    if "g2_probe_lds" in s:
        return
    anchor = "inline long g2_env_long(const char* name, long dflt) {"
    s = s.replace(anchor, HELPER + "\n" + anchor)
    f.write_text(s)


def sub_file(name, pairs):
    f = G2 / name
    s = f.read_text()
    for old, new, count in pairs:
        if new in s:
            continue
        assert old in s, f"{name}: not found: {old[:70]}"
        s = s.replace(old, new, count)
    f.write_text(s)


add_helper()

sub_file("g2_wu.cuh", [
    ("hipLaunchKernelGGL(g2::g2_wu_kernel<true>, dim3(T / 64, H), dim3(256), 0, s, p);",
     'hipLaunchKernelGGL(g2::g2_wu_kernel<true>, dim3(T / 64, H), dim3(256), g2::g2_probe_lds(g2::g2_wu_kernel<true>, "KDA_PROBE_LDS_WU", 0), s, p);', 1),
    ("hipLaunchKernelGGL(g2::g2_wu_kernel<false>, dim3(T / 64, H), dim3(256), 0, s, p);",
     'hipLaunchKernelGGL(g2::g2_wu_kernel<false>, dim3(T / 64, H), dim3(256), g2::g2_probe_lds(g2::g2_wu_kernel<false>, "KDA_PROBE_LDS_WU", 0), s, p);', 1),
])

sub_file("g2_dav.cuh", [
    ("hipLaunchKernelGGL(g2::g2_dav_kernel<true>, dim3(T / 64, H), dim3(256), 0, s, p);",
     'hipLaunchKernelGGL(g2::g2_dav_kernel<true>, dim3(T / 64, H), dim3(256), g2::g2_probe_lds(g2::g2_dav_kernel<true>, "KDA_PROBE_LDS_DAV", 0), s, p);', 1),
    ("hipLaunchKernelGGL(g2::g2_dav_kernel<false>, dim3(T / 64, H), dim3(256), 0, s, p);",
     'hipLaunchKernelGGL(g2::g2_dav_kernel<false>, dim3(T / 64, H), dim3(256), g2::g2_probe_lds(g2::g2_dav_kernel<false>, "KDA_PROBE_LDS_DAV", 0), s, p);', 1),
])

sub_file("g2_bwd_tail.cuh", [
    ("hipLaunchKernelGGL(g2::g2_bwd_tail_kernel<true>, dim3(NT, H), dim3(256), 0, s, p);",
     'hipLaunchKernelGGL(g2::g2_bwd_tail_kernel<true>, dim3(NT, H), dim3(256), g2::g2_probe_lds(g2::g2_bwd_tail_kernel<true>, "KDA_PROBE_LDS_TAIL", 0), s, p);', 1),
    ("hipLaunchKernelGGL(g2::g2_bwd_tail_kernel<false>, dim3(NT, H), dim3(256), 0, s, p);",
     'hipLaunchKernelGGL(g2::g2_bwd_tail_kernel<false>, dim3(NT, H), dim3(256), g2::g2_probe_lds(g2::g2_bwd_tail_kernel<false>, "KDA_PROBE_LDS_TAIL", 0), s, p);', 1),
])

sub_file("g2_wy.cuh", [
    ('hipLaunchKernelGGL((p3_wy_s4a_kernel<true, true>), dim3(NT, H), dim3(256), 0, s, pa);',
     'hipLaunchKernelGGL((p3_wy_s4a_kernel<true, true>), dim3(NT, H), dim3(256), g2::g2_probe_lds(p3_wy_s4a_kernel<true, true>, "KDA_PROBE_LDS_S4A", 0), s, pa);', 1),
    ('hipLaunchKernelGGL((p3_wy_s4a_kernel<true, false>), dim3(NT, H), dim3(256), 0, s, pa);',
     'hipLaunchKernelGGL((p3_wy_s4a_kernel<true, false>), dim3(NT, H), dim3(256), g2::g2_probe_lds(p3_wy_s4a_kernel<true, false>, "KDA_PROBE_LDS_S4A", 0), s, pa);', 1),
    ('hipLaunchKernelGGL((p3_wy_s4a_kernel<false, true>), dim3(NT, H), dim3(256), 0, s, pa);',
     'hipLaunchKernelGGL((p3_wy_s4a_kernel<false, true>), dim3(NT, H), dim3(256), g2::g2_probe_lds(p3_wy_s4a_kernel<false, true>, "KDA_PROBE_LDS_S4A", 0), s, pa);', 1),
    ('hipLaunchKernelGGL((p3_wy_s4a_kernel<false, false>), dim3(NT, H), dim3(256), 0, s, pa);',
     'hipLaunchKernelGGL((p3_wy_s4a_kernel<false, false>), dim3(NT, H), dim3(256), g2::g2_probe_lds(p3_wy_s4a_kernel<false, false>, "KDA_PROBE_LDS_S4A", 0), s, pa);', 1),
    ('hipLaunchKernelGGL(p3_wy_s4b_kernel<true>, dim3(NT, H), dim3(256), 0, s, pb);',
     'hipLaunchKernelGGL(p3_wy_s4b_kernel<true>, dim3(NT, H), dim3(256), g2::g2_probe_lds(p3_wy_s4b_kernel<true>, "KDA_PROBE_LDS_S4B", 0), s, pb);', 1),
    ('hipLaunchKernelGGL(p3_wy_s4b_kernel<false>, dim3(NT, H), dim3(256), 0, s, pb);',
     'hipLaunchKernelGGL(p3_wy_s4b_kernel<false>, dim3(NT, H), dim3(256), g2::g2_probe_lds(p3_wy_s4b_kernel<false>, "KDA_PROBE_LDS_S4B", 0), s, pb);', 1),
])

sub_file("g2_fwd_prep.cuh", [
    ('hipLaunchKernelGGL((g2::r4::g2_fwd_prep_a_kernel<1, true>), grid, blk, 0, s, pa);',
     'hipLaunchKernelGGL((g2::r4::g2_fwd_prep_a_kernel<1, true>), grid, blk, g2::g2_probe_lds(g2::r4::g2_fwd_prep_a_kernel<1, true>, "KDA_PROBE_LDS_PREPA", 0), s, pa);', 1),
    ('hipLaunchKernelGGL((g2::r4::g2_fwd_prep_a_kernel<1, false>), grid, blk, 0, s, pa);',
     'hipLaunchKernelGGL((g2::r4::g2_fwd_prep_a_kernel<1, false>), grid, blk, g2::g2_probe_lds(g2::r4::g2_fwd_prep_a_kernel<1, false>, "KDA_PROBE_LDS_PREPA", 0), s, pa);', 1),
    ('hipLaunchKernelGGL((g2::r4::g2_fwd_prep_a_kernel<0, true>), grid, blk, 0, s, pa);',
     'hipLaunchKernelGGL((g2::r4::g2_fwd_prep_a_kernel<0, true>), grid, blk, g2::g2_probe_lds(g2::r4::g2_fwd_prep_a_kernel<0, true>, "KDA_PROBE_LDS_PREPA", 0), s, pa);', 1),
    ('hipLaunchKernelGGL((g2::r4::g2_fwd_prep_a_kernel<0, false>), grid, blk, 0, s, pa);',
     'hipLaunchKernelGGL((g2::r4::g2_fwd_prep_a_kernel<0, false>), grid, blk, g2::g2_probe_lds(g2::r4::g2_fwd_prep_a_kernel<0, false>, "KDA_PROBE_LDS_PREPA", 0), s, pa);', 1),
    ('hipLaunchKernelGGL(g2::r4::g2_fwd_prep_b_kernel<true>, grid, blk, 0, s, pb);',
     'hipLaunchKernelGGL(g2::r4::g2_fwd_prep_b_kernel<true>, grid, blk, g2::g2_probe_lds(g2::r4::g2_fwd_prep_b_kernel<true>, "KDA_PROBE_LDS_PREPB", 0), s, pb);', 1),
    ('hipLaunchKernelGGL(g2::r4::g2_fwd_prep_b_kernel<false>, grid, blk, 0, s, pb);',
     'hipLaunchKernelGGL(g2::r4::g2_fwd_prep_b_kernel<false>, grid, blk, g2::g2_probe_lds(g2::r4::g2_fwd_prep_b_kernel<false>, "KDA_PROBE_LDS_PREPB", 0), s, pb);', 1),
])

# fwd_v8: probe overrides the P-24 throttle fallback
s = (G2 / "g2_fwd_v8.cuh").read_text()
old = "const size_t dyn = throttle ? g2_lds_topup(kern, 24 * 1024) : 0;"
new = 'const size_t dyn = g2_probe_lds(kern, "KDA_PROBE_LDS_FWH", throttle ? g2_lds_topup(kern, 24 * 1024) : 0);'
if new not in s:
    assert s.count(old) == 2, f"fwd_v8 dyn sites: {s.count(old)}"
    s = s.replace(old, new)
    (G2 / "g2_fwd_v8.cuh").write_text(s)

print("patched:", ROOT)
