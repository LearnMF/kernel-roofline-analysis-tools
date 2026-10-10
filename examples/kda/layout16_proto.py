"""Prototype patch (TIMING ONLY, not numerically valid): make v8o4 / dhu v8b / dhu v8b4 read
their per-lane 8 B operand fragments as 16 B pairs from a hypothetical paired layout
(adjacent 256-element sub-blocks interleaved per lane).  Applied to a COPY of csrc/g2 for
the standalone layout16_bench.hip; inputs are random fill, so only the kernel time matters.
Usage: python3 layout16_proto.py <copy>/csrc/g2"""
import sys
from pathlib import Path

d = Path(sys.argv[1])
common = d / "g2_common.h"
s = common.read_text()
helper = """__device__ __forceinline__ f4 ld16f(const float* p) { return *reinterpret_cast<const f4*>(p); }
typedef short v8bh_ __attribute__((ext_vector_type(8)));
// PROTO: one 16 B load = two adjacent 8 B fragments of the paired layout
__device__ __forceinline__ void ld16x2(const bf16* p, v4bh& a, v4bh& b) {
    const v8bh_ x = *reinterpret_cast<const v8bh_*>(p);
    a = __builtin_shufflevector(x, x, 0, 1, 2, 3);
    b = __builtin_shufflevector(x, x, 4, 5, 6, 7);
}"""
s = s.replace("__device__ __forceinline__ f4 ld16f(const float* p) { return *reinterpret_cast<const f4*>(p); }", helper, 1)
common.write_text(s)


def rep(path, old, new, count=1):
    t = path.read_text()
    assert t.count(old) >= count, (path.name, old[:70], t.count(old))
    path.write_text(t.replace(old, new))


v8 = d / "g2_fwd_v8.cuh"
t = v8.read_text()
k0 = t.index("g2_fwd_h_v8o4_kernel(FwdOParams p)")
k1 = t.index("\n}\n", k0)
body = t[k0:k1]
subs = [
    ("""        const bf16* src = p.Wn + (((chunk + i) * 4 + r1) * 8) * 256 + l * 4;
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt) wf[kt] = ld8(src + kt * 256);""",
     """        const bf16* src = p.Wn + (((chunk + i) * 4 + r1) * 8) * 256 + l * 8;
        #pragma unroll
        for (int j = 0; j < 4; ++j) ld16x2(src + j * 512, wf[2 * j], wf[2 * j + 1]);"""),
    ("""        const bf16* src = p.QgA + (((chunk + i) * 4 + r1) * 8) * 256 + l * 4;
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt) qgf[kt] = ld8(src + kt * 256);""",
     """        const bf16* src = p.QgA + (((chunk + i) * 4 + r1) * 8) * 256 + l * 8;
        #pragma unroll
        for (int j = 0; j < 4; ++j) ld16x2(src + j * 512, qgf[2 * j], qgf[2 * j + 1]);"""),
    ("""        const bf16* src = p.AqkA + (((chunk + i) * 4 + r1) * 4) * 256 + l * 4;
        #pragma unroll
        for (int st = 0; st < 4; ++st) aqf[st] = ld8(src + st * 256);""",
     """        const bf16* src = p.AqkA + (((chunk + i) * 4 + r1) * 4) * 256 + l * 8;
        #pragma unroll
        for (int j = 0; j < 2; ++j) ld16x2(src + j * 512, aqf[2 * j], aqf[2 * j + 1]);"""),
    ("""        uf = ld8(p.Un + (((chunk + i) * 4 + r1) * 8 + 2 * vs + v1a_) * 256 + l * 4);""",
     """        if (v1a_ == 0) ld16x2(p.Un + (((chunk + i) * 4 + r1) * 8 + 2 * vs) * 256 + l * 8, uf0, uf1);
        (void)uf;"""),
    ("""        for (int rt = 0; rt < 4; ++rt)
            kf[rt] = ld8(p.Kn + (((chunk + i) * 8 + kb) * 4 + rt) * 256 + l * 4);""",
     """        for (int j = 0; j < 2; ++j)
            ld16x2(p.Kn + (((chunk + i) * 8 + kb) * 4) * 256 + j * 512 + l * 8, kf[2 * j], kf[2 * j + 1]);"""),
]
for old, new in subs:
    assert body.count(old) == 1, ("v8o4", old[:60], body.count(old))
    body = body.replace(old, new)
v8.write_text(t[:k0] + body + t[k1:])

eng = d / "g2_engine.cuh"
t = eng.read_text()
ka_old = """        const bf16* src = p.KgA + (((chunk + i) * 4 + r1) * 8) * 256 + l * 4;
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt) ka[kt] = ld8(src + kt * 256);"""
ka_new = """        const bf16* src = p.KgA + (((chunk + i) * 4 + r1) * 8) * 256 + l * 8;
        #pragma unroll
        for (int j = 0; j < 4; ++j) ld16x2(src + j * 512, ka[2 * j], ka[2 * j + 1]);"""
for name in ("g2_dhu_v8b_kernel(BwdNParams p)", "g2_dhu_v8b4_kernel(BwdNParams p)"):
    k0 = t.index(name)
    k1 = t.index("\n}\n", k0)
    body = t[k0:k1]
    assert body.count(ka_old) == 1, (name, "ka")
    body = body.replace(ka_old, ka_new)
    if "v8b4" in name:
        old = """        for (int rt = 0; rt < 4; ++rt) {
            qb[rt] = ld8(p.Qg + (((chunk + i) * 8 + kb) * 4 + rt) * 256 + l * 4);
            wb[rt] = ld8(p.Wb + (((chunk + i) * 8 + kb) * 4 + rt) * 256 + l * 4);
        }"""
        new = """        for (int j = 0; j < 2; ++j) {
            ld16x2(p.Qg + (((chunk + i) * 8 + kb) * 4) * 256 + j * 512 + l * 8, qb[2 * j], qb[2 * j + 1]);
            ld16x2(p.Wb + (((chunk + i) * 8 + kb) * 4) * 256 + j * 512 + l * 8, wb[2 * j], wb[2 * j + 1]);
        }"""
    else:
        old = """        for (int rt = 0; rt < 4; ++rt) {
            qb[rt] = ld8(p.Qg + (((chunk + i) * 8 + wv) * 4 + rt) * 256 + l * 4);
            wb[rt] = ld8(p.Wb + (((chunk + i) * 8 + wv) * 4 + rt) * 256 + l * 4);
        }"""
        new = """        for (int j = 0; j < 2; ++j) {
            ld16x2(p.Qg + (((chunk + i) * 8 + wv) * 4) * 256 + j * 512 + l * 8, qb[2 * j], qb[2 * j + 1]);
            ld16x2(p.Wb + (((chunk + i) * 8 + wv) * 4) * 256 + j * 512 + l * 8, wb[2 * j], wb[2 * j + 1]);
        }"""
    assert body.count(old) == 1, (name, "qw", body.count(old))
    body = body.replace(old, new)
    t = t[:k0] + body + t[k1:]
eng.write_text(t)
print("PROTO PATCHED")
