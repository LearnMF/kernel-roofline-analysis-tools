"""kra P-60 (bitwise): forward recurrence (MODE 1, chain-bound TP8 instance) issues QgA one step ahead.

Critical-path analysis of the production MODE 1 step (cp_step.py, SQTT 16K/12): ~1150-1400 of ~3600 cycles per
step are the latency of the QgA load, which is issued right after barrier A and consumed ~150 instructions later by
the oh MMACs (nothing else covers it).  QgA/AqkA do not depend on the recurrence, so the PF instance issues
loadQg(i+1) right after this step's oh pass (qgf's last use) and loadAq(i+1) right after this step's o pass (aqf's
last use): the same registers, now in flight across ~2000 cycles of the next step instead of ~1100.
Arithmetic and operand order unchanged -> bit-identical.  Used for MODE 1 only in the SG (<= 80 CTAs) regime.
    python3 p60_qg_prefetch.py <tree>"""
import sys
from pathlib import Path

f = Path(sys.argv[1]) / "csrc" / "g2" / "g2_fwd_v8.cuh"
s = f.read_text()


def sub(s, old, new, cnt=1):
    assert s.count(old) == cnt, (old[:80], s.count(old))
    return s.replace(old, new)


sig = "template <bool VL, int MODE = 0, bool GE = true, bool SG = false>"
s = sub(s, sig, sig[:-1] + ", bool PF = false>")
head, rest = s.split("g2_fwd_h_v8o4_kernel(FwdOParams p) {", 1)
body, tail = rest.split("// kra P-49: g2_fwd_h_v8o4_kernel with NVH", 1)
SB = "if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);"
body = sub(body, """    write_h(kb0, ht0); write_h(kb1, ht1);
    for (int i = c0; i < c1; ++i) {""", """    write_h(kb0, ht0); write_h(kb1, ht1);
    if constexpr (PF) { loadQg(c0); loadAq(c0); }   // kra P-60: QgA/AqkA one step ahead
    for (int i = c0; i < c1; ++i) {""")
body = sub(body, """        if (hasO || hasOb) {         // R3: issue this step's""",
           """        if (!PF && (hasO || hasOb)) {         // R3: issue this step's""")
body = sub(body, """                oh1 = mmac(qgf[kt], hbf1[kt], oh1);
            }
        }
""", f"""                oh1 = mmac(qgf[kt], hbf1[kt], oh1);
            }}
        }}
        if constexpr (PF) {{ {SB} loadQg(nxt); {SB} }}   // kra P-60: qgf's last use was the oh pass
""")
body = sub(body, """        // ---- GEMM2 for both K-blocks""", f"""        if constexpr (PF) {{ {SB} loadAq(nxt); {SB} }}   // kra P-60: aqf's last use was the o pass
        // ---- GEMM2 for both K-blocks""")
s = head + "g2_fwd_h_v8o4_kernel(FwdOParams p) {" + body + "// kra P-49: g2_fwd_h_v8o4_kernel with NVH" + tail
s = sub(s, "case 1: if (sg) launch4(g2_fwd_h_v8o4_kernel<VL, 1, true, true>);",
        "case 1: if (sg) launch4(g2_fwd_h_v8o4_kernel<VL, 1, true, true, true>);   // kra P-60")
f.write_text(s)
print("P-60 APPLIED")
