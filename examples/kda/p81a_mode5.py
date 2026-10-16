"""kra P-81a probe (bitwise): forward recurrence MODE 5 = production MODE 1 (Ob) + bN(h) emit, no Vn / aN -- the
forward half of option d (results/kda-segfusion-20261016 §8: save h, fold v_new = U - W h into wu, drop the backward
recompute recurrence).  Instance <VL, 5, GE, SG, PF, KP, VP, BP> (MODE 1's production template plus BP's bN-from-
registers emit, as MODE 2 combines VP+BP); selected for Ob + HbN without Vn/ANv at <= 80 CTAs with the kp layouts.
    python3 p81a_mode5.py <tree>"""
import sys
from pathlib import Path

f = Path(sys.argv[1]) / "csrc" / "g2" / "g2_fwd_v8.cuh"
s = f.read_text()


def sub(b, old, new, cnt=1):
    assert b.count(old) == cnt, (old[:90], b.count(old))
    return b.replace(old, new)


s = sub(s, """    const bool hasVn = MODE == 0 ? p.Vn != nullptr : MODE != 1;
    const bool hasAN = MODE == 0 ? p.ANv != nullptr : (MODE != 1 && MODE != 4);""",
        """    const bool hasVn = MODE == 0 ? p.Vn != nullptr : (MODE != 1 && MODE != 5);   // kra P-81a: MODE 5 = Ob + HbN
    const bool hasAN = MODE == 0 ? p.ANv != nullptr : (MODE != 1 && MODE != 4 && MODE != 5);""", cnt=s.count("""    const bool hasVn = MODE == 0 ? p.Vn != nullptr : MODE != 1;
    const bool hasAN = MODE == 0 ? p.ANv != nullptr : (MODE != 1 && MODE != 4);"""))
s = sub(s, """    else if (O == nullptr && Ob == nullptr && Vn && HbN && !ANv) mode = 4;   // kra P-22""",
        """    else if (O == nullptr && Ob == nullptr && Vn && HbN && !ANv) mode = 4;   // kra P-22
    else if (O == nullptr && Ob != nullptr && !Vn && HbN && !ANv) mode = 5;   // kra P-81a""")
s = sub(s, """            case 1: if (sg) { if (kp) launch4(g2_fwd_h_v8o4_kernel<VL, 1, true, true, true, true, true>);   // kra P-60""",
        """            case 5: if (sg && kp) { launch4(g2_fwd_h_v8o4_kernel<VL, 5, true, true, true, true, true, true>); break; }   // kra P-81a
                    launch4(g2_fwd_h_v8o4_kernel<VL, 0>); break;
            case 1: if (sg) { if (kp) launch4(g2_fwd_h_v8o4_kernel<VL, 1, true, true, true, true, true>);   // kra P-60""")
f.write_text(s)
print("P-81a APPLIED")
