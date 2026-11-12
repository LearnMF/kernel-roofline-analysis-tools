"""Patterns for lds_m_bank.hip (write: python3 lds_m_pat.py > lds_m_pat.txt) and the PMC decoder
(python3 lds_m_pat.py decode pmc.csv R CTAS)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../results/kda-gfx938-ldsbank-20261103"))


def patterns():
    P = []
    P.append(("contig 16l", [16 * l for l in range(64)]))
    P.append(("stride 256", [256 * l for l in range(64)]))
    P.append(("stride 272", [272 * l for l in range(64)]))
    P.append(("bcast 0", [0] * 64))
    P.append(("16-lane groups same 256 B", [16 * (l % 16) for l in range(64)]))
    P.append(("16-lane groups +2048", [16 * (l % 16) + 2048 * (l // 16) for l in range(64)]))
    P.append(("16-lane groups +2064", [16 * (l % 16) + 2064 * (l // 16) for l in range(64)]))
    for x in range(1, 64):   # lane 0 and lane x on one bank slot (different rows), every other lane on one address
        a = [4096 + 16] * 64
        a[0], a[x] = 0, 2048
        P.append((f"pair 0,{x}", a))
    import lds_model as M
    def wu(ph, PJ, PN, n=4):
        return [a for (w, a) in M.wu_phases(0, PJ, PN)[ph][:n]]
    for i, a in enumerate(wu("w5 u-MMAC ds_read_m32x16", 68, 136, 2)): P.append((f"base w5 PN136 #{i}", a))
    for i, a in enumerate(wu("w5 u-MMAC ds_read_m32x16", 64, 128, 2)): P.append((f"base w5 PN128 #{i}", a))
    for i, a in enumerate(wu("w7 HbN ds_read_m32x16", 68, 136, 2)): P.append((f"base w7 #{i}", a))
    M.hbn_idx = lambda sp, k, v: sp * 4096 + k * 32 + ((((v >> 3) ^ (((k >> 1) & 1) << 1) ^ (((k >> 2) ^ (k >> 4)) & 1))) << 3) + (v & 7)
    def nat(j, v, PN):
        X = (((j >> 4) & 1) << 2) if PN == 136 else ((j & 3) | (((j >> 4) & 1) << 2))
        return j * PN + ((((v >> 3) ^ X) & 15) << 3) + (v & 7)
    M.nat = nat
    for i, a in enumerate(wu("w5 u-MMAC ds_read_m32x16", 68, 136, 2)): P.append((f"bc3 w5 PN136 #{i}", a))
    for i, a in enumerate(wu("w5 u-MMAC ds_read_m32x16", 64, 128, 2)): P.append((f"bc3 w5 PN128 #{i}", a))
    for i, a in enumerate(wu("w7 HbN ds_read_m32x16", 68, 136, 2)): P.append((f"bc3 w7 #{i}", a))
    for name, a in P:
        assert len(a) == 64 and all(x % 16 == 0 and 0 <= x < 32768 for x in a), name
    return P


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "decode":
        import csv
        rows = list(csv.DictReader(open(sys.argv[2])))
        rows = [r for r in rows if "probe" in r["KernelName"]]
        R, C = int(sys.argv[3]), int(sys.argv[4])
        P = patterns()
        k = 0
        print(f"{'pattern':28s} | conflict cycles per instruction: ds_read_m32x16 | ds_read_b128 | ds_read_b64")
        for name, _ in P:
            v = []
            for op in range(3):
                r0, r1 = rows[k], rows[k + 1]; k += 2
                f = lambda r, c: float(r[c] or 0)
                v.append((f(r1, "SQ_LDS_BANK_CONFLICT") - f(r0, "SQ_LDS_BANK_CONFLICT")) / (C * R))
                v.append((f(r1, "SQ_LDS_IDX_ACTIVE") - f(r0, "SQ_LDS_IDX_ACTIVE")) / (C * R))
            print(f"{name:28s} | m {v[0]:5.2f} (active {v[1]:5.2f}) | b128 {v[2]:5.2f} ({v[3]:5.2f}) | b64 {v[4]:5.2f} ({v[5]:5.2f})")
    else:
        for name, a in patterns():
            print(" ".join(map(str, a)))
