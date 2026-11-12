"""Random / exhaustive search of GF(2)-linear XOR swizzles per wu layout on the fitted bank model."""
import collections, itertools, random, lds_model as M
random.seed(1)

def lin(cols):            # cols[b] = output mask contributed by input bit b
    def f(x):
        o = 0
        for b, c in enumerate(cols):
            if (x >> b) & 1: o ^= c
        return o
    return f

def cost_phases(phs, PJ, PN):
    tot = collections.Counter()
    for wv in range(4):
        P = M.wu_phases(wv, PJ, PN)
        for ph in phs:
            for w, a in P[ph]: tot[ph[:2]] += M.cost(w, a)
    return tot

names = {k[:2]: k for k in M.wu_phases(0).keys()}
def search(tag, setter, nbits, outmask_choices, phs, PJ, PN, n=1500, exhaustive=False):
    best = None
    cands = itertools.product(outmask_choices, repeat=nbits) if exhaustive else (
        tuple(random.choice(outmask_choices) for _ in range(nbits)) for _ in range(n))
    for cols in cands:
        setter(lin(cols))
        t = cost_phases([names[p] for p in phs], PJ, PN); s = sum(t.values())
        if best is None or s < best[0]: best = (s, cols, dict(t))
    print(f"{tag}: best {best}")
    return best

import sys
which = sys.argv[1:]
o4 = list(range(16)); o3 = list(range(8)); o2 = list(range(4))
if "hbn" in which:     # granule' = (v>>3) ^ f(k), f 2-bit, k bits 0..6
    def set_h(f): M.hbn_idx = lambda sp, k, v: sp * 4096 + k * 32 + (((v >> 3) ^ f(k)) << 3) + (v & 7)
    search("hbn", set_h, 7, o2, ["w6", "w7"], 68, 136, exhaustive=True)
if "tile" in which:    # granule ^= f(e >> 6), f 4-bit, e bits 6..12
    def set_t(f): M.tile_idx = lambda e: (((e >> 2) ^ f((e >> 6) & 127)) << 2) | (e & 3)
    search("tile", set_t, 7, o4, ["w2", "w3", "w8", "w9"], 68, 136, n=3000)
if "slab64" in which:
    def set_s(f): M.slab_col = lambda r, j: ((((j >> 2) ^ f(r)) & 15) << 2) | (j & 3)
    search("slab PJ64", set_s, 7, o4, ["w0", "w1"], 64, 128, n=4000)
if "nat128" in which:  # natural rows: granule (v>>3) ^ X(j), X 4-bit, j bits 0..5
    def set_n(f): M.nat = lambda j, v, PN: j * PN + ((((v >> 3) ^ f(j)) & 15) << 3) + (v & 7)
    search("nat PN128", set_n, 6, o4, ["w4", "w5"], 64, 128, n=4000)
if "nat144" in which:
    def set_n(f): M.nat = lambda j, v, PN: j * PN + ((((v >> 3) ^ f(j)) & 15) << 3) + (v & 7)
    search("nat PN144", set_n, 6, o4, ["w4", "w5"], 68, 144, n=2000)
if "nat136" in which:
    def set_n(f): M.nat = lambda j, v, PN: j * PN + ((((v >> 3) ^ f(j)) & 15) << 3) + (v & 7)
    search("nat PN136", set_n, 6, o4, ["w4", "w5"], 68, 136, n=2000)
