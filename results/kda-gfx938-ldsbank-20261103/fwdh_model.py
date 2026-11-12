"""fwd_h wide kernel (g2_fwd_h_v8o4w_kernel, NVH = 4, production at H >= 64 on gfx938) LDS phases on the bank model."""
import collections, lds_model as M
kPadK, kPadT, NVH = 132, 68, 4
pi16 = lambda n: 4 * (n & 3) + (n >> 2)
def phases(wv):
    P = collections.OrderedDict(); L = range(64); m = lambda l: l & 15; q4 = lambda l: l >> 4
    ins = []
    for kb in (wv, wv + 4):
        for vt in range(NVH):
            for e in range(4):
                ins.append((2, [2 * ((16 * vt + m(l)) * kPadK + 16 * kb + q4(l) + 4 * e) for l in L]))
    P["f0 write_h (b16 st)"] = ins
    ins = []
    for kb in (wv, wv + 4):
        for vt in range(NVH):
            for e in range(4):
                ins.append((2, [2 * ((16 * vt + 4 * q4(l) + e) * kPadK + 16 * kb + pi16(m(l))) for l in L]))
    P["f1 bN_emit gathers (b16 ld)"] = ins
    ins = [(8, [2 * ((16 * v + m(l)) * kPadK + 16 * kt + 4 * q4(l)) for l in L]) for kt in range(8) for v in range(NVH)]
    P["f2 GEMM1 hB reads (b64)"] = ins
    P["f4 o hB reads (b64)"] = ins
    ins = []
    for v in range(NVH):
        for e in range(4):
            ins.append((2, [2 * ((16 * v + q4(l) + 4 * e) * kPadT + 16 * wv + m(l)) for l in L]))
    P["f3 vB writes (b16 st)"] = ins
    ins = [(8, [2 * ((16 * v + m(l)) * kPadT + 16 * st + 4 * q4(l)) for l in L]) for v in range(NVH) for st in range(4)]
    P["f5 o vB reads (b64)"] = ins
    P["f7 update vB reads (b64) x2"] = ins + ins
    base = 2 * 64 * kPadK + 2 * 64 * kPadT + wv * 512   # oStage after hB / vB (bytes)
    ins = []
    for v in range(NVH):
        for e in range(4):
            ins.append((2, [base + 2 * (m(l) * 16 + q4(l) + 4 * e) for l in L]))
        ins.append((8, [base + 2 * (m(l) * 16 + 4 * q4(l)) for l in L]))
    P["f6 oStage (b16 st + b64 ld)"] = ins
    return P
tot = collections.Counter(); n = collections.Counter()
for wv in range(4):
    for ph, ins in phases(wv).items():
        n[ph] += len(ins)
        for w, a in ins: tot[ph] += M.cost(w, a)
for ph in n: print(f"{ph:32s} insts {n[ph]:4d}  extra cycles / CTA-step {tot[ph]:5d}")
print("total per CTA-step", sum(tot.values()))
