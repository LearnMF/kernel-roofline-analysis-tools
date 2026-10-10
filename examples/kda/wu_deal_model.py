"""Offline access model of g2_wu phase 1/2 lane "deals" (which lane handles which (row j,
column quad vq) item in which iteration).  For each deal it reports, per wave instruction:
  - store line coverage of the vB/kgB emit (and KgA): 128 B lines touched and bytes per line
    (a line written completely by one instruction = 128 B; partial lines cost extra, P-16c),
  - load coalescing of the row-major k/v reads (bytes used per 128 B line touched),
  - LDS bank conflicts of the kbT staging writes (max lanes on one 4 B bank; 1 = conflict-free
    for b16 scalar writes that share a dword only within a lane pair),
so a deal can be judged before touching the kernel.   python3 wu_deal_model.py"""
import collections

KPADJ2 = 68


def pi16(n):
    return 4 * (n & 3) + (n >> 2)


def vb_pos(j, vq):             # vB / kgB layout, quad base inside the 8192-element block
    return 4 * pi16(j & 15) + 64 * ((vq & 15) >> 2) + 256 * (vq >> 4) + 2048 * (j >> 4)


def nat16(o):
    return (o & ~511) | (((o >> 2) & 63) << 3) | (((o >> 8) & 1) << 2) | (o & 3)


def kga_pos(j, vq):            # KgA: A form, plain row nibble, then the L16 pairing
    return nat16(4 * (j & 15) + 64 * ((vq & 15) >> 2) + 256 * (vq >> 4) + 2048 * (j >> 4))


def slab_col(r, j):
    return ((((j >> 2) ^ (r >> 3)) & 15) << 2) | (j & 3)


def deal_base(tid, it):        # production: t = tid + 256*it
    t = tid + 256 * it
    return t >> 5, (t & 31) << 2


def deal_pair(tid, it):        # P-16c: rows j, j+4 in iterations 2g, 2g+1
    r = tid >> 5
    return 16 * (it >> 1) + (r & 3) + 8 * ((r >> 2) & 1) + 4 * (it & 1), (tid & 31) << 2


def deal_line(tid, it):        # P-16d: wave w = row group, lane l = 8a + b; b -> row pair base,
    w, l = tid >> 6, tid & 63  # a + 8s -> vq pair (vq, vq+16); it = 4s + 2x + y: x = vq half, y = row
    a, b = l >> 3, l & 7
    s, x, y = it >> 2, (it >> 1) & 1, it & 1
    c = a + 8 * s
    vq = 4 * (c & 3) + 32 * (c >> 2) + 16 * x
    j = 16 * w + (b & 3) + 8 * (b >> 2) + 4 * y
    return j, vq


def report(name, deal, pair_rows):
    seen = set()
    st_lines, st_bpl, kga_bpl, ld_bpl, lds_worst = [], [], [], [], 0
    for it in range(8):
        for w in range(4):
            lanes = [deal(64 * w + l, it) for l in range(64)]
            seen.update((it, w, l) for l in range(64))
            # store: vB/kgB.  Pairing deals emit on odd iterations: 16 B = rows (j-4, j) at vq
            if pair_rows is None or pair_rows(it):
                c = collections.Counter()
                for j, vq in lanes:
                    jj = j - 4 if pair_rows is not None else j
                    nbytes = 16 if pair_rows is not None else 8
                    base = 2 * vb_pos(jj, vq)
                    for b0 in range(base, base + nbytes, 8):
                        c[b0 // 128] += 8
                st_lines.append(len(c)); st_bpl.append(sum(c.values()) / len(c))
            c = collections.Counter()
            for j, vq in lanes:
                c[(2 * kga_pos(j, vq)) // 128] += 8
            kga_bpl.append(sum(c.values()) / len(c))
            # load: row-major [T, H*128] bf16; one head's row segment is 256 B
            c = collections.Counter()
            for j, vq in lanes:
                c[(j * 4096 + 2 * vq) // 128] += 8   # row stride irrelevant beyond line identity
            ld_bpl.append(sum(c.values()) / len(c))
            # LDS kbT staging: 4 scalar b16 writes per lane (dv = 0..3), bank = dword % 32
            for dv in range(4):
                c = collections.Counter()
                for j, vq in lanes:
                    vv = vq + dv
                    sr = 16 * (vv >> 4) + pi16(vv & 15)
                    addr = 2 * (sr * KPADJ2 + slab_col(sr, j))
                    c[(addr // 4) % 32] += 1
                lds_worst = max(lds_worst, max(c.values()))
    items = {deal(tid, it) for tid in range(256) for it in range(8)}
    print(f"{name:10s} items {len(items)}/2048 | vB/kgB store: {sum(st_lines)/len(st_lines):5.1f} lines/instr, "
          f"{sum(st_bpl)/len(st_bpl):5.1f} B/line | KgA store {sum(kga_bpl)/len(kga_bpl):5.1f} B/line | "
          f"load {sum(ld_bpl)/len(ld_bpl):5.1f} B/line | kbT LDS worst {lds_worst} lanes/bank")


report("base", deal_base, None)
report("P-16c", deal_pair, lambda it: it & 1)
report("P-16d", deal_line, lambda it: it & 1)
