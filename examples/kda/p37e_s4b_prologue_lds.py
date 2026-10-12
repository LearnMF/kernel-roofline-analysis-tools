"""kra P-37e (round 12, s4b rewrite step 2c, A-class): stage the s4b prologue's dA / dAqk tiles
through LDS instead of per-lane scattered gathers.

SQTT of P-37d (1 CTA/CU): prologue-type waits (sites hit <= 2x per wave) are 22.9% of wave-cycles
vs 9.9% inside the kt loop -- at one CTA per CU nothing hides the CTA start.  The prologue reads
the dA handoff and dAqk 64x64 fp32 tiles twice: rows (dak, aqA: f4 loads) and TRANSPOSED columns
(akB, aqB: up to 32 scalar loads per lane, every one touching 64 cache lines).  Both tiles are now
fetched with coalesced 16 B buffer_load ... lds (4 per thread each) into LDS that is idle during
the prologue (dA -> the double table area, 26 KB; dAqk -> stage 1, 18 KB; masked dAqk rows land as
zeros), then every operand is read from LDS.  Stage 1 (tile kt=1) is issued after the gathers, so
the kt=0 barrier waits vmcnt(0) for it.  Values unchanged.
Apply after p37 + p37c + p37d.        python3 p37e_s4b_prologue_lds.py <tree>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wy.cuh"
t = P.read_text()
a = t.index("// ================================================================ kra P-37: pipelined s4b")
b = t.index("p3_wy_s4b_pipe_kernel(WyS4BParams p) {", a)
seg = t[a:b]


def rep(old, new, count=1):
    global seg
    assert seg.count(old) == count, (old[:80], seg.count(old))
    seg = seg.replace(old, new)


# 1) issue only stage 0 up front, plus the two prologue tiles
rep("""    s4bp_issue<A>(io, sm.st[0], 0);
    s4bp_issue<A>(io, sm.st[1], 1);
""", """    s4bp_issue<A>(io, sm.st[0], 0);
    float* const lA = reinterpret_cast<float*>(&sm.TA[0][0]);   // kra P-37e: dA tile (prologue only)
    float* const lQ = reinterpret_cast<float*>(&sm.st[1]);      //            dAqk tile
    {
        const g2_v4i rdA = s4bp_srd(dAg, (unsigned)(kBT * RA * 4));
        const g2_v4i rAq = s4bp_srd(aq_, (unsigned)(kBT * RA * 4));
        #pragma unroll
        for (int it_ = 0; it_ < 4; ++it_) {
            const int u = it_ * 256 + tid, r_ = u >> 4, c_ = u & 15;
            const unsigned off = (unsigned)((r_ * RA + 4 * c_) * 4);
            s4bp_ld(rdA, off, lA + (it_ * 256 + A * 64) * 4);
            s4bp_ld(rAq, g2_row_ok<M>(r_, cl) ? off : kS4bpOOB, lQ + (it_ * 256 + A * 64) * 4);
        }
    }
""")
# 2) move the dak / aqA / akB / aqB block after the retire+barrier and read it from LDS
old_block_start = "    f4 dak[4];\n    #pragma unroll\n    for (int ut = 0; ut < 4; ++ut) dak[ut] = ld16f(dAg + row * RA + 16 * ut + 4 * q);"
old_block_end = "            aqB[b - A][x] = (keep && g2_row_ok<M>(j, cl)) ? aq_[j * RA + row] : 0.f;\n        }\n"
i0 = seg.index(old_block_start)
i1 = seg.index(old_block_end, i0) + len(old_block_end)
seg = seg[:i0] + seg[i1:]
rep("""    #pragma unroll
    for (int b = 0; b < 4; ++b) asm volatile("" :: "v"(dak[b]));
    #pragma unroll
    for (int b = 0; b <= A; ++b) asm volatile("" :: "v"(aqA[b]));
    #pragma unroll
    for (int b = A; b < 4; ++b) asm volatile("" :: "v"(aqB[b - A]), "v"(akB[b - A]));
""", "")
rep("""    s4bp_vmcnt<0>();
    lds_barrier();                                         // stage 0 (and 1) visible
""", """    s4bp_vmcnt<0>();
    lds_barrier();                                         // stage 0 + dA / dAqk tiles visible
    f4 dak[4];
    #pragma unroll
    for (int ut = 0; ut < 4; ++ut) dak[ut] = *reinterpret_cast<const f4*>(&lA[row * 64 + 16 * ut + 4 * q]);
    f4 aqA[A + 1], aqB[4 - A], akB[4 - A];
    #pragma unroll
    for (int b = 0; b <= A; ++b) {
        aqA[b] = rok ? *reinterpret_cast<const f4*>(&lQ[row * 64 + 16 * b + 4 * q]) : f4{0.f, 0.f, 0.f, 0.f};
        if (b == A) {
            #pragma unroll
            for (int x = 0; x < 4; ++x)
                if (4 * q + x > m) aqA[b][x] = 0.f;
        }
    }
    #pragma unroll
    for (int b = A; b < 4; ++b)
        #pragma unroll
        for (int x = 0; x < 4; ++x) {
            const int j = 16 * b + 4 * q + x;
            const bool keep = b > A || 4 * q + x >= m;
            akB[b - A][x] = keep ? lA[j * 64 + row] : 0.f;    // dAT[row][j] == dA[j][row]
            aqB[b - A][x] = (keep && g2_row_ok<M>(j, cl)) ? lQ[j * 64 + row] : 0.f;
        }
    lds_barrier();                                         // table area + stage 1 free again
    s4bp_issue<A>(io, sm.st[1], 1);
""")
# 3) the kt=0 barrier now waits for stage 1 (issued right before the loop)
rep("        if (kt > 0 && kt + 1 < 8) s4bp_vmcnt<3>();",
    "        if (kt == 0) s4bp_vmcnt<0>();                    // kra P-37e: stage 1 issued after the prologue\n        else if (kt + 1 < 8) s4bp_vmcnt<3>();")
t = t[:a] + seg + t[b:]
P.write_text(t)
print("P-37e (prologue dA/dAqk via LDS) APPLIED")
