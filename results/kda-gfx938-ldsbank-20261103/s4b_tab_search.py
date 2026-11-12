import itertools, lds_model as M
def cost(tix):
    t = 0
    for A in range(4):
        for e in range(4):
            t += M.cost(4, [4 * tix(4 * e + (l >> 4), 16 * A + (l & 15)) for l in range(64)])
        for b in range(4):   # reads: every block (TA b<=A, TQ/TK b>=A: each block is read by some wave)
            t += M.cost(16, [4 * tix(l & 15, 16 * b + 4 * (l >> 4)) for l in range(64)])
    return t
best = None
for cols in itertools.product(range(16), repeat=4):
    def h(r, cols=cols):
        o = 0
        for i in range(4):
            if (r >> i) & 1: o ^= cols[i]
        return o
    tix = lambda r, c, h=h: r * 64 + ((((c >> 2) ^ h(r)) & 15) << 2) + (c & 3)
    c = cost(tix)
    if best is None or c < best[0]:
        best = (c, cols)
        if c == 0: break
print("best", best, "TB(r&15):", cost(lambda r, c: r * 64 + ((((c >> 2) ^ (r & 15))) << 2) + (c & 3)), "TS68:", cost(lambda r, c: r * 68 + c))
