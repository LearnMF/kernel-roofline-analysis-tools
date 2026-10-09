"""LDS access patterns of g2_wu_kernel (R5) and candidate layouts, through kra.lds.bankmodel.
All counts are per wave (256-thread CTA = 4 waves; wave w = threads 64w..64w+63)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kra.lds.bankmodel import pattern_cost  # noqa: E402

pi16 = lambda n: 4 * (n & 3) + (n >> 2)
WAVE = 0     # pattern is identical for every wave up to a constant offset


def tid(i, l, stride=256):
    return i * stride + 64 * WAVE + l


def model(PAD_SLAB=68, PAD_SA=64, swz_slab=None, swz_tile=None):
    """swz_*: f(row) -> xor applied to the 4-short granule index of the column."""
    sx = swz_slab or (lambda r: 0)
    tx = swz_tile or (lambda e: e)
    slab = lambda r, j: 2 * (r * PAD_SLAB + ((((j >> 2) ^ sx(r)) << 2) | (j & 3)))
    res = {}
    # A. sA fill: b64 writes, 4 per lane
    res["sA_fill_w64"] = pattern_cost(lambda i, l: 2 * ((tid(i, l) >> 4) * PAD_SA + ((tid(i, l) & 15) << 2)), 4, 8)
    # B. phase-1 fill: 8 iters x 4 b16 writes, row = 16*(vv>>4)+pi16(vv&15), col j
    def p1(i, l, dv):
        t = tid(i // 4, l); j, vq = t >> 5, (t & 31) << 2; vv = vq + dv
        return slab(16 * (vv >> 4) + pi16(vv & 15), j)
    res["p1_fill_w16"] = pattern_cost(lambda i, l: p1(i, l, i % 4), 32, 2)
    # C. MMAC phase (x2): per js: 1 sA b64 read + 8 slab b64 reads
    w = WAVE
    res["mmac_sA_r64"] = pattern_cost(
        lambda i, l: 2 * ((16 * w + (l & 15)) * PAD_SA + 16 * (i % 4) + 4 * (l >> 4)), 8, 8)
    res["mmac_slab_r64"] = pattern_cost(
        lambda i, l: slab(16 * ((i // 4) % 8) + (l & 15), 16 * (i % 4) + 4 * (l >> 4)), 64, 8)
    # D. phase-2 fill: 8 iters x 4 b16 writes, row = vq+dv
    def p2(i, l, dv):
        t = tid(i // 4, l); j, vq = t >> 5, (t & 31) << 2
        return slab(vq + dv, j)
    res["p2_fill_w16"] = pattern_cost(lambda i, l: p2(i, l, i % 4), 32, 2)
    # E. phase-3 tile: 8 iters x 4 b16 writes at ibn+4dv ; 8 linear b64 reads
    def p3(i, l, dv):
        t = tid(i // 4, l); j, vq = t >> 5, (t & 31) << 2
        ibn = (j & 3) + 4 * (vq & 15) + 64 * ((j >> 2) & 3) + 256 * (j >> 4) + 1024 * (vq >> 4)
        return 2 * tx(ibn + 4 * dv)
    res["p3_tile_w16"] = pattern_cost(lambda i, l: p3(i, l, i % 4), 32, 2)
    res["p3_tile_r64"] = pattern_cost(lambda i, l: 2 * tx(4 * tid(i, l)), 8, 8)
    tot = {k: sum(v[k] for v in res.values()) for k in ("cost", "ideal", "conflict")}
    return res, tot


def show(name, **kw):
    res, tot = model(**kw)
    print(f"== {name}: conflict cycles/wave {tot['conflict']}, cost {tot['cost']} (ideal {tot['ideal']})")
    for k, v in res.items():
        print(f"   {k:14s} cost {v['cost']:5d} ideal {v['ideal']:4d} conflict {v['conflict']:5d} degree {v['degree']:.2f}")
    return tot


if __name__ == "__main__":
    base = show("R5 baseline")
