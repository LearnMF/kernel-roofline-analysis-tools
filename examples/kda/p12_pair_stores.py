"""kra P-12: the last two L16 producers with 8 B stores into the paired layout (each touches
twice the cache lines of the plain layout, vmem_store) -> full 16 B pair stores via a lane
exchange (the two halves of a pair sit in partner lanes of the same wave).
  * prep_a QgA: sub-block 8rb + 2wv + (q4 >> 1); the partner is lane l ^ 32 (same m, same
    slot).  Lane l < 32 writes the qg[0] pair, lane l >= 32 the qg[1] pair: all lanes store.
  * wu KgA: sub-block (vq >> 4) + 8(j >> 4); the partner is lane l ^ 4 (vq +- 16, same j,
    same slot); the even lane of each pair writes 16 B.
Values and positions unchanged -> bit-identical.  Usage: python3 p12_pair_stores.py <g2 dir>"""
import sys
from pathlib import Path

K = Path(sys.argv[1])


def edit(path, old, new):
    t = path.read_text()
    assert t.count(old) == 1, (path.name, old[:80], t.count(old))
    path.write_text(t.replace(old, new))


XCH = """__device__ __forceinline__ v4bh lane_xor_v4bh(v4bh x, int mask) {   // kra P-12
    const int2 v = *reinterpret_cast<const int2*>(&x);
    int2 r;
    r.x = __shfl_xor(v.x, mask, 64);
    r.y = __shfl_xor(v.y, mask, 64);
    return *reinterpret_cast<const v4bh*>(&r);
}
"""
edit(K / "g2_common.h", "// store both halves of a pair at once", XCH + "// store both halves of a pair at once")

edit(K / "g2_fwd_prep.cuh",
     """            *reinterpret_cast<v4bh*>(p.QgA + nat16(qb0)) = qg[0];
            *reinterpret_cast<v4bh*>(p.QgA + nat16(qb0 + 64)) = qg[1];""",
     """            {   // kra P-12: lanes l and l^32 hold the two halves of the QgA pairs (sub-blocks
                // 2(4rb + wv) + (q4 >> 1)); l < 32 stores the qg[0] pair, l >= 32 the qg[1] pair
                const bool lo = q4 < 2;
                const v4bh mine = lo ? qg[0] : qg[1];
                const v4bh other = lane_xor_v4bh(lo ? qg[1] : qg[0], 32);   // partner's same slot
                // even half (sub-block 2c) always comes from the lane with q4 < 2
                const v4bh ev = lo ? mine : other, od = lo ? other : mine;
                const size_t qe = qb0 - (size_t)256 * ((cbase >> 4) & 1) + (lo ? 0 : 64);
                st16x2(p.QgA + nat16(qe), ev, od);
            }""")
edit(K / "g2_wu.cuh",
     """            *reinterpret_cast<v4bh*>(p.KgA + nat16(nblk + (size_t)(4 * (j & 15)
                + 64 * ((vq & 15) >> 2) + 256 * (vq >> 4) + 2048 * (j >> 4)))) =
                *reinterpret_cast<const v4bh*>(gkv);""",
     """            {   // kra P-12: lanes l and l^4 hold the two halves of a KgA pair (vq +- 16, same
                // slot); the lane with the even sub-block writes the full 16 B
                const v4bh mine = *reinterpret_cast<const v4bh*>(gkv);
                const v4bh other = lane_xor_v4bh(mine, 4);
                if (((vq >> 4) & 1) == 0)
                    st16x2(p.KgA + nat16(nblk + (size_t)(4 * (j & 15)
                        + 64 * ((vq & 15) >> 2) + 256 * (vq >> 4) + 2048 * (j >> 4))), mine, other);
            }""")
print("P-12 APPLIED")
