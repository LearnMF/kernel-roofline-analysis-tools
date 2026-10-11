"""kra P-34b (round 11, on top of P-34a; bitwise-identical; found with SQTT after P-34): two of the
unrolled staging loops still carry a per-iteration `if (ptr != nullptr)` GLOBAL store that the
production route never takes (dav's aN_do since P-20, wu's vB since P-17).  A possibly-aliasing
global store between the loads keeps the compiler from hoisting load i+1 above store i, so SQTT
still shows dav's do loop as 8 serialized load -> vmcnt(0) round trips (25.9% of wave-cycles).
The null test is uniform per launch: test it once outside and give the production branch a copy of
the loop without the store.      Usage: python3 p34b_hoist_dead_store.py <tree root>"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) / "csrc" / "g2"


def patch(f, loop, store, cond):
    p = ROOT / f
    t = p.read_text()
    assert t.count(loop) == 1, (f, loop[:70], t.count(loop))
    assert store in loop
    fast = loop.replace(store, "")
    t = t.replace(loop, f"        if ({cond} == nullptr) {{   // kra P-34b: production route, no dead store in the loop\n"
                        + fast + "        } else {\n" + loop + "        }\n")
    p.write_text(t)


DAV_LOOP = """        #pragma unroll   // kra P-34: all loads in flight before the first wait
        for (int it_ = 0; it_ < 8; ++it_) {
            const int t = tid + 256 * it_;
            const int j = t >> 5, vq = (t & 31) << 2;
            const v4bh dv_ = g2_row_ok<M>(j, cl)
                ? *reinterpret_cast<const v4bh*>(p.do_ + vbase + (size_t)j * rstride + vq)
                : v4bh{0, 0, 0, 0};
            #pragma unroll
            for (int d = 0; d < 4; ++d)
                slab[(size_t)(vq + d) * kPadJ3 + j] = dv_[d];
            // T3: the SAME do loads also emit P2's dOT (B form, stride-4 quads, std
            // block order) and P3's aN_do (A form, plain row nibble, P3 block order)
            // -- the two B3.5 convert2 calls (which re-read do, ~200 MB @8K/48)
            // disappear.  Index math identical to g2_wu's verified Kn/KgA emits.
            if (p.aNdo != nullptr)   // kra P-20: production route -- s4b reads do row-major
                *reinterpret_cast<v4bh*>(p.aNdo + nblk3 + (size_t)(4 * (j & 15)
                    + 64 * ((vq & 15) >> 2) + 256 * (vq >> 4) + 2048 * (j >> 4))) = dv_;
        }
"""
DAV_STORE = """            if (p.aNdo != nullptr)   // kra P-20: production route -- s4b reads do row-major
                *reinterpret_cast<v4bh*>(p.aNdo + nblk3 + (size_t)(4 * (j & 15)
                    + 64 * ((vq & 15) >> 2) + 256 * (vq >> 4) + 2048 * (j >> 4))) = dv_;
"""
WU_LOOP = """        #pragma unroll   // kra P-34: all loads in flight before the first wait
        for (int it_ = 0; it_ < 8; ++it_) {
            const int t = tid + 256 * it_;
            const int j = t >> 5, vq = (t & 31) << 2;
            const size_t so = sbase + (size_t)j * rstride + vq;
            const v4bh vv = g2_row_ok<M>(j, cl) ? *reinterpret_cast<const v4bh*>(p.v + so)
                                                 : v4bh{0, 0, 0, 0};
            const float b = sBeta[j];
            #pragma unroll
            for (int dv = 0; dv < 4; ++dv)
                slab[(size_t)(vq + dv) * kPadJ2 + wu_slab_col(vq + dv, j)] = bf_rne(bf2f(vv[dv]) * b);
            if (p.vB != nullptr)   // kra P-17: production route reads v row-major in s4a instead
                *reinterpret_cast<v4bh*>(p.vB + wu_vb_quad(j, vq, nblk3)) = vv;
        }
"""
WU_STORE = """            if (p.vB != nullptr)   // kra P-17: production route reads v row-major in s4a instead
                *reinterpret_cast<v4bh*>(p.vB + wu_vb_quad(j, vq, nblk3)) = vv;
"""
patch("g2_dav.cuh", DAV_LOOP, DAV_STORE, "p.aNdo")
patch("g2_wu.cuh", WU_LOOP, WU_STORE, "p.vB")
print("P-34b APPLIED")
