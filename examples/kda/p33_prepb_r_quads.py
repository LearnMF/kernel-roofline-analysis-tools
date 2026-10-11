"""kra P-33 (round 10, bitwise-identical): the fp32 scratch R (written by fwd_prep_a, read ONLY by
fwd_prep_b) is copied out of prep_a's LDS verbatim, so prep_b's R4(b) gathers each lane's quad
{r[0], r[64], r[128], r[192]} with FOUR 4 B loads (VMEM-issue bound kernel, 80%; wave I issues
4 * (2 + 2I) of them -> 32 for the critical wave 3).  prep_a now writes R lane-quad major --
element (b, e, x = 4m + q4) at b*256 + 4x + e instead of b*256 + 64e + x -- gathering the 4 values
from LDS (stride 64 floats: one bank per e, conflict-free) for each coalesced 16 B store, and
prep_b's R4 becomes ONE 16 B load.  Same values -> bit-identical.  (fwd_intra2's own R is separate.)
Usage: python3 p33_prepb_r_quads.py <tree root>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_fwd_prep.cuh"
t = P.read_text()


def sub(old, new):
    global t
    assert t.count(old) == 1, (old[:80], t.count(old))
    t = t.replace(old, new)


sub("""    {
        const f4* s4 = reinterpret_cast<const f4*>(smem);
        f4* d4 = reinterpret_cast<f4*>(p.R + g2_nat_blk(i, hh, p.H, p.NT, false, 5120));
        #pragma unroll
        for (int j = threadIdx.x; j < 1280; j += 256) d4[j] = s4[j];
    }
    });   // g2_vl_dispatch""",
    """    {   // kra P-33: R lane-quad major -- out quad j = (b = j >> 6, x = j & 63) holds the LDS
        // elements b*256 + 64e + x, e = 0..3 (R4's quad), so prep_b reads it with one 16 B load
        f4* d4 = reinterpret_cast<f4*>(p.R + g2_nat_blk(i, hh, p.H, p.NT, false, 5120));
        #pragma unroll
        for (int j = threadIdx.x; j < 1280; j += 256) {
            const float* src = smem + (j >> 6) * 256 + (j & 63);
            d4[j] = f4{src[0], src[64], src[128], src[192]};
        }
    }
    });   // g2_vl_dispatch""")
sub("""    auto R4 = [&](int b) {
        const float* r = Rg + b * 256 + 4 * m + q4;
        return f4{r[0], r[64], r[128], r[192]};
    };""",
    """    auto R4 = [&](int b) {   // kra P-33: the quad is contiguous (lane-quad major R)
        return *reinterpret_cast<const f4*>(Rg + b * 256 + 4 * (4 * m + q4));
    };""")
P.write_text(t)
print("P-33 APPLIED")
