"""kra P-70c probe (on top of P-70 + P-70b): only the tail prep is dispatched chunk-major (KDA_G2_PIPE_TCM, default 1).
P-70b trace (K=48, masked): with the head-major tail, head 11's first tail chunk is ready only at the end of the tail
prep, so every head-11 CTA of the recurrence waits (629 vs 518 us).  The head stays head-major (chunk-major costs
prep_a +20% / prep_b +63% over a full grid).
    python3 p70c_tailcm.py <tree>"""
import sys
from pathlib import Path

b = Path(sys.argv[1]) / "csrc" / "g2" / "g2_bindings.cpp"
s = b.read_text()
for old, new in [
    ("        g2_prep_range_set(off, n, g2_pipe_cm(), fl, 1);   // chunk-major only on request (it slows prep)",
     "        g2_prep_range_set(off, n, fl != nullptr ? g2_pipe_tcm() : g2_pipe_cm(), fl, 1);   // kra P-70c: tail chunk-major"),
    ('static int g2_pipe_cm() { const char* e = getenv("KDA_G2_PIPE_CM"); return e ? atoi(e) : 0; }',
     'static int g2_pipe_cm() { const char* e = getenv("KDA_G2_PIPE_CM"); return e ? atoi(e) : 0; }\n'
     'static int g2_pipe_tcm() { const char* e = getenv("KDA_G2_PIPE_TCM"); return e ? atoi(e) : 1; }'),
]:
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)
b.write_text(s)
print("P-70c APPLIED")
