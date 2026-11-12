"""gfx938 full-chain ledger from hipprof --pmc memory / util CSVs (kbench --all --iters 2: the last call of each kernel).
python3 ledger.py memory.csv util.csv SHAPE"""
import csv, sys, re
SHORT = [("prep_a", "prep_a"), ("prep_b", "prep_b"), ("fwd_h", "fwd_h"), ("gate", "g2_gate"), ("wu", "g2_wu"),
         ("dav", "g2_dav"), ("dhu", "g2_dhu"), ("s4a", "s4a"), ("s4b", "s4b"), ("tail", "g2_bwd_tail_kernel"),
         ("tail_red", "bwd_tail_reduce")]
def load(fn):
    rows = {}
    for r in csv.DictReader(open(fn)):
        for nm, key in SHORT:
            if key in r["KernelName"] and (nm != "tail" or "reduce" not in r["KernelName"]):
                rows[nm] = r                      # last occurrence wins (second iteration)
    return rows
mem, utl, shape = load(sys.argv[1]), load(sys.argv[2]), sys.argv[3]
f = lambda r, k: float(r.get(k) or 0)
print(f"### {shape}\n| kernel | time us | HBM read MB | HBM write MB | HBM GB/s | VALU inst / CU-cycle | VMEM inst / CU-cycle | LDS bank conflict / LDS active | lds B | vgpr |")
print("|---|---|---|---|---|---|---|---|---|---|")
tot_t = tot_r = tot_w = 0
for nm, _ in SHORT:
    if nm not in mem: continue
    m, u = mem[nm], utl.get(nm, {})
    t = (f(m, "EndNs") - f(m, "BeginNs")) / 1e3
    rd = (f(m, "TCC_EA_RDREQ_sum") - f(m, "TCC_EA_RDREQ_32B_sum")) * 64 + f(m, "TCC_EA_RDREQ_32B_sum") * 32
    wr = f(m, "TCC_EA_WRREQ_64B_sum") * 64 + (f(m, "TCC_EA_WRREQ_sum") - f(m, "TCC_EA_WRREQ_64B_sum")) * 32
    cyc = f(u, "SQ_BUSY_CU_CYCLES") or 1
    lc = f(m, "SQ_LDS_BANK_CONFLICT") / max(f(m, "SQ_LDS_IDX_ACTIVE"), 1)
    tot_t += t; tot_r += rd; tot_w += wr
    print(f"| {nm} | {t:.1f} | {rd/1e6:.1f} | {wr/1e6:.1f} | {(rd+wr)/t/1e3:.0f} | {f(u,'SQ_ACTIVE_INST_VALU')/cyc:.3f} | "
          f"{f(u,'SQ_ACTIVE_INST_VMEM')/cyc:.3f} | {lc:.3f} | {m['lds']} | {m['arch_vgpr']} |")
print(f"| **total** | {tot_t:.1f} | {tot_r/1e6:.1f} | {tot_w/1e6:.1f} | {(tot_r+tot_w)/tot_t/1e3:.0f} | | | | | |")
