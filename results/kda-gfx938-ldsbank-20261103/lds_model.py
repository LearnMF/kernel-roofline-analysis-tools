"""LDS address model of g2_dav / g2_wu (gfx938 production instances on 2631309) per LDS phase.

Every phase is a list of wave instructions; an instruction = (width in bytes, [byte address of each of the 64 lanes]).
conflicts(): extra cycles per instruction under a simple bank model (32 banks x 4 B; lanes served in groups of
128 / width lanes per cycle; within a group the cost is the largest number of distinct dwords on one bank).
python3 lds_model.py [phase ...]"""
import sys, collections

pi16 = lambda n: 4 * (n & 3) + (n >> 2)


# ---------------- dav (gfx938: unpadded 16 KB slab, XOR swizzle) ----------------
kK = 128
i1 = lambda r, c: r * kK + (c ^ ((r & 15) << 2))
i2 = lambda v, j: v * 64 + (j ^ ((v & 15) << 2))
iA = lambda t, j: t * 64 + (j ^ ((t & 15) << 2))     # fp32 index


def dav_phases(wv):
    P = collections.OrderedDict()
    L = range(64)
    ins = []
    for it in range(8):
        for e in range(4):
            a = []
            for l in L:
                t = 64 * wv + l + 256 * it; sb, c = t >> 6, t & 63
                d0, d1, m, q = sb >> 3, sb & 7, c & 15, c >> 4
                a.append(2 * i1(16 * d0 + m, 16 * d1 + q + 4 * e))
            ins.append((2, a))
    P["d0 pass-1 Vn staging (b16 st)"] = ins
    ins = []
    for it in range(4):
        for h in range(2):
            a = []
            for l in L:
                t = 64 * wv + l + 256 * it; g, c = t >> 6, t & 63
                aa, vt = g >> 2, 2 * (g & 3); r, cc = 16 * aa + (c & 15), 4 * (c >> 4)
                a.append(2 * i1(r, 16 * vt + 16 * h + cc))
            ins.append((8, a))
    P["d1 VnA re-emit reads (b64 ld)"] = ins
    ins = []
    for vs in range(8):
        for jt in range(4):
            ins.append((8, [2 * i1(16 * jt + (l & 15), 16 * vs + (l >> 4) * 4) for l in L]))
    P["d2 pass-1 MMAC B reads (b64 ld)"] = ins
    ins = []
    for jt in range(4):
        for e in range(4):
            a = []
            for l in L:
                t, j0 = 16 * wv + (l & 15), 16 * jt + (l >> 4)
                a.append(4 * iA(t, j0 + 4 * e))
            ins.append((4, a))
    P["d3 dA fp32 staging (b32 st)"] = ins
    ins = []
    for it in range(4):
        a = []
        for l in L:
            c = 64 * wv + l + 256 * it; t, j = c >> 4, (c & 15) * 4
            a.append(4 * iA(t, j))
        ins.append((16, a))
    P["d4 dA staging reads (b128 ld)"] = ins
    ins = []
    for it in range(8):
        for d in range(4):
            a = []
            for l in L:
                t = 64 * wv + l + 256 * it; j, vq = t >> 5, (t & 31) << 2
                a.append(2 * i2(vq + d, j))
            ins.append((2, a))
    P["d5 pass-2 do scatter (b16 st)"] = ins
    ins = []
    for it in range(4):
        for h in range(2):
            a = []
            for l in L:
                c = 64 * wv + l + 256 * it; x = 8 * c
                j0 = 4 * ((x >> 6) & 3) + 16 * ((x >> 8) & 3); v0 = ((x >> 2) & 15) + 16 * (x >> 10)
                a.append(2 * i2(v0 + h, j0))
            ins.append((8, a))
    P["d6 dOT reads (b64 ld)"] = ins
    ins = []
    for js in range(4):
        for vt in range(8):
            ins.append((8, [2 * i2(16 * vt + (l & 15), 16 * js + (l >> 4) * 4) for l in L]))
    P["d7 pass-2 MMAC B reads (b64 ld)"] = ins
    return P


# ---------------- wu (gfx938 production <false, true, true, 0>: padded slab PJ 68, natural rows PN 136) ----------------
PJ, PN = 68, 136
slab_col = lambda r, j: ((((j >> 2) ^ (r >> 3)) & 15) << 2) | (j & 3)
tile_idx = lambda e: (((e >> 2) ^ ((e >> 10) & 7)) << 2) | (e & 3)
hbn_idx = lambda sp, k, v: sp * 4096 + k * 32 + ((((v >> 3) ^ (((k >> 1) & 1) << 1))) << 3) + (v & 7)


def mls_row(l, jp):
    o, g = l & 14, l >> 4
    ie = 4 + (o >> 2) if (o & 2) else (o >> 2)
    return 16 * (2 * jp + (ie >> 2)) + 4 * g + (ie & 3)


def ibn_of(t):
    j, vq = t >> 5, (t & 31) << 2
    return (j & 3) + 4 * (vq & 15) + 64 * ((j >> 2) & 3) + 256 * (j >> 4) + 1024 * (vq >> 4)


nat = lambda j, v, PN: j * PN + v          # natural vb row index (fill2 / u-MMAC MLS reads)


def wu_phases(wv, PJ=PJ, PN=PN):
    P = collections.OrderedDict()
    L = range(64)
    ins = []
    for it in range(8):
        for dv in range(4):
            a = []
            for l in L:
                t = 64 * wv + l + 256 * it; j, vq = t >> 5, (t & 31) << 2
                vv = vq + dv; sr = 16 * (vv >> 4) + pi16(vv & 15)
                a.append(2 * (sr * PJ + slab_col(sr, j)))
            ins.append((2, a))
    P["w0 fill1 kbT (b16 st)"] = ins
    ins = []
    for js in range(4):
        for vt in range(8):
            a = []
            for l in L:
                br = 16 * vt + (l & 15)
                a.append(2 * (br * PJ + slab_col(br, 16 * js + (l >> 4) * 4)))
            ins.append((8, a))
    P["w1 w-MMAC B reads (b64 ld)"] = ins
    ins = []
    for vt in range(8):
        for e in range(4):
            a = []
            for l in L:
                ib = ((l & 15) & 3) + 16 * (l >> 4) + 64 * ((l & 15) >> 2) + 256 * wv + 1024 * vt
                a.append(2 * tile_idx(ib + 4 * e))
            ins.append((2, a))
    P["w2 Wb tile (b16 st)"] = ins
    emit = []
    for k in range(4):
        for h in range(2):
            a = []
            for l in L:
                t = 64 * wv + l + 256 * k; o0 = 512 * (t >> 6) + 4 * (t & 63)
                a.append(2 * tile_idx(o0 + 256 * h))
            emit.append((8, a))
    P["w3 Wb emit16 reads (b64 ld)"] = emit
    ins = []
    for it in range(8):
        a = []
        for l in L:
            t = 64 * wv + l + 256 * it; j, vq = t >> 5, (t & 31) << 2
            a.append(2 * nat(j, vq, PN))
        ins.append((8, a))
    P["w4 fill2 natural vb (b64 st)"] = ins
    ins = []
    for jp in range(2):
        for vt in range(8):
            ins.append(("m", [2 * nat(mls_row(l, jp), 8 * (l & 1) + 16 * vt, PN) for l in L]))
    P["w5 u-MMAC ds_read_m32x16"] = ins
    ins = []
    for pr in range(2):
        for it in range(8):
            a = []
            for l in L:
                jj = 64 * wv + l + 256 * it
                ln, vt, kb, sp = jj & 63, (jj >> 6) & 1, (jj >> 7) & 7, jj >> 10
                qq, mm = ln >> 4, ln & 15
                a.append(2 * hbn_idx(sp, 16 * kb + pi16(mm), 16 * vt + 4 * qq))
            ins.append((8, a))
    P["w6 HbN staging (b64 st)"] = ins
    ins = []
    for pr in range(2):
        for sp in range(2):
            for jp in range(4):
                for h in range(2):
                    ins.append(("m", [2 * hbn_idx(sp, mls_row(l, jp), 16 * h + 8 * (l & 1)) for l in L]))
    P["w7 HbN ds_read_m32x16"] = ins
    ins = []
    for it in range(8):
        for dv in range(4):
            a = [2 * tile_idx(ibn_of(64 * wv + l + 256 * it) + 4 * dv) for l in L]
            ins.append((2, a))
    P["w8 Qn tile (b16 st)"] = ins
    P["w9 Qn emit16 reads (b64 ld)"] = emit
    return P


def groups(width):
    """lanes served together per LDS cycle on gfx938: b16 / b32 32 lanes, b64 16 (lane order); b128 and ds_read_m32x16
    (width "m") 8 lanes each, grouped by lane bits as measured with examples/kda/gfx938/lds_m_bank.hip"""
    if width == "m":     # measured (lds_m_bank.hip pair probes): lanes with equal (b1 ^ b4, b3, b5) share a cycle
        return [[l for l in range(64) if (((l >> 1) ^ (l >> 4)) & 1, (l >> 3) & 1, l >> 5) == key]
                for key in [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]]
    if width == 16:      # ds_read_b128, measured: lanes with equal (b2 ^ b4, b3, b5)
        return [[l for l in range(64) if (((l >> 2) ^ (l >> 4)) & 1, (l >> 3) & 1, l >> 5) == key]
                for key in [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]]
    per = 32 if width <= 4 else 128 // width
    return [list(range(g0, g0 + per)) for g0 in range(0, 64, per)]


def cost(width, addrs, half_merge=True):
    """extra cycles of one wave instruction: per lane group, the largest number of distinct dwords on one bank, minus 1
    (half_merge: two b16 lanes on the two halves of one dword do not conflict -- what gfx938 measures)"""
    ext = 0
    nb = 16 if width == "m" else max(width, 4)
    for g in groups(width):
        banks = collections.defaultdict(set)
        for a in (addrs[l] for l in g):
            for d in range(nb // 4):
                dw = (a >> 2) + d
                key = dw if half_merge or width != 2 else (dw, a & 2)
                banks[dw & 31].add(key)
        ext += max(len(s) for s in banks.values()) - 1
    return ext


if __name__ == "__main__":
    for name, fn in (("dav", dav_phases), ("wu", wu_phases)):
        tot = collections.Counter(); n = collections.Counter()
        for wv in range(4):
            for ph, ins in fn(wv).items():
                n[ph] += len(ins)
                for w, a in ins:
                    tot[ph] += cost(w, a)
                    tot[ph + " [b16 halves split]"] += (cost(w, a, False) if w == 2 else 0)
        print(f"== {name} (per CTA: 4 waves)")
        for ph in n:
            print(f"  {ph:36s} insts {n[ph]:4d}  model extra cycles {tot[ph]:5d}  (b16 halves as conflict: {tot[ph + ' [b16 halves split]'] or '-'})")
