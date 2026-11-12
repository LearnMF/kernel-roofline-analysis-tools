"""wu LDS layouts on the fitted bank model: base (2631309), P-938bc3 (first try, wrong ds_read_m grouping assumed) and
P-938bc4 (kept candidate), both gfx938 instances (padded PJ 68 / PN 136 and L16 PJ 64 / PN 128).
Measured per CTA (hipprof): base 4096 / 8704, bc3 3328 / 2304, bc4 1280 / 768 -- the model reproduces all six."""
import collections, importlib, lds_model as M


def evalw(PJ, PN):
    tot = collections.Counter()
    for wv in range(4):
        for ph, ins in M.wu_phases(wv, PJ, PN).items():
            for w, a in ins: tot[ph[:2]] += M.cost(w, a)
    return dict(tot), sum(tot.values())


def bc3(L16):
    def slab_col(r, j):
        h = (((r >> 4) & 7) << 1) if not L16 else ((r & 15) ^ (((r >> 4) & 3) << 2))
        return ((((j >> 2) ^ h) & 15) << 2) | (j & 3)
    M.slab_col = slab_col
    M.tile_idx = lambda e: (((e >> 2) ^ ((e >> 10) & 7) ^ ((e >> 6) & 3)) << 2) | (e & 3)
    M.hbn_idx = lambda sp, k, v: sp * 4096 + k * 32 + ((((v >> 3) ^ (((k >> 1) & 1) << 1) ^ (((k >> 2) ^ (k >> 4)) & 1))) << 3) + (v & 7)
    M.nat = lambda j, v, PN: j * PN + ((((v >> 3) ^ ((((j >> 4) & 1) << 2) if PN == 136 else ((j & 3) | (((j >> 4) & 1) << 2)))) & 15) << 3) + (v & 7)


def bc4(L16):
    bc3(L16)
    M.hbn_idx = lambda sp, k, v: sp * 4096 + k * 32 + ((((v >> 3) ^ ((((k >> 1) ^ (k >> 2)) & 1) << 1) ^ ((k >> 3) & 1))) << 3) + (v & 7)
    M.nat = lambda j, v, PN: j * PN + ((((v >> 3) ^ ((((j & 1) << 1) | (((j >> 2) & 1) << 2)) if PN == 128 else 0)) & 15) << 3) + (v & 7)


for name, setup in (("base", None), ("bc3", bc3), ("bc4", bc4)):
    for L16, PJ, PN in ((False, 68, 136), (True, 64, 128)):
        importlib.reload(M)
        if setup: setup(L16)
        d, s = evalw(PJ, PN)
        print(f"{name:4s} {'L16' if L16 else 'pad'}: {d} sum {s}")
