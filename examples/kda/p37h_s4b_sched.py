"""kra P-37h (round 12, scheme 2: scheduling of the 1-CTA pipelined s4b; A-class).  Variants:
  h1  deferred stores: tile kt's three fp32 stores are issued in iteration kt+1 right after the
      pass-K MMAC chain (SQTT: VMEM_WR issue delay 3.5% when they sit back to back before the
      barrier).  vmcnt bookkeeping unchanged (stores(kt-1) are still the only ops issued after
      stage kt+1's group when the kt barrier waits).
  h2  spread async loads: the stage loads of tile kt+2 are issued one by one between the back
      half's sub-phases instead of back to back after the barrier (VMEM_RD issue delay 6.3%).
  h3  branch-free LDS table writes: cs2 (msum: all 16 lanes of the row hold the same sum), cs1
      (xsum16: all 4 lanes of the column hold it) and gM (= g8, which every lane of the wave holds)
      are written by every lane -- same value, same address -- and the rlast dg correction is a
      select; removes the exec-mask branches that cut the loop into small scheduling regions.
Apply on a tree with p37 + p37c + p37d + p37e + p37f.   python3 p37h_s4b_sched.py <tree> h1,h2,h3"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wy.cuh"
want = set(sys.argv[2].split(","))
t = P.read_text()
a = t.index("// ================================================================ kra P-37: pipelined s4b")
b = t.index("p3_wy_s4b_pipe_kernel(WyS4BParams p) {", a)
seg = t[a:b]


def rep(old, new, count=1):
    global seg
    assert seg.count(old) == count, (old[:90], seg.count(old))
    seg = seg.replace(old, new)


if "h1" in want:
    rep("    float dbacc = 0.f;\n    #pragma unroll 1\n    for (int kt = 0; kt < 8; ++kt) {",
        "    float dbacc = 0.f;\n    f4 poq{}, pok{}, pog{};                               // kra P-37h1: deferred stores\n"
        "    unsigned pso = kS4bpOOB;\n    #pragma unroll 1\n    for (int kt = 0; kt < 8; ++kt) {")
    rep("        {\n            float hd = 0.f;",
        "        if (kt > 0) s4bp_st3(io.rdq, io.rdk, io.rdg, pso, poq, pok, pog);   // tile kt-1 (P-37h1)\n"
        "        {\n            float hd = 0.f;")
    rep("        const unsigned so = rok ? (unsigned)((ro + 16 * kt) * 4) : kS4bpOOB;\n"
        "        s4bp_st3(io.rdq, io.rdk, io.rdg, so, oq, ok, og);",
        "        pso = rok ? (unsigned)((ro + 16 * kt) * 4) : kS4bpOOB;\n"
        "        poq = oq; pok = ok; pog = og;")
    rep("    db = xsum16(db) + xsum16(dbacc);\n    if (q == 0 && rok) p.db[(size_t)(tok0 + row) * H + hh] = db;",
        "    s4bp_st3(io.rdq, io.rdk, io.rdg, pso, poq, pok, pog);  // tile 7 (P-37h1)\n"
        "    db = xsum16(db) + xsum16(dbacc);\n    if (q == 0 && rok) p.db[(size_t)(tok0 + row) * H + hh] = db;")

if "h2" in want:
    rep("        if (kt + 2 < 8) s4bp_issue<A>(io, S, kt + 2);      // tile kt+2 -> the stage kt freed\n",
        "        const bool nx_ = kt + 2 < 8;                       // kra P-37h2: tile kt+2, spread out\n"
        "        const unsigned k2_ = (unsigned)(kt + 2);\n"
        "        if (nx_) s4bp_ld(io.rh, io.oh + k2_ * 4096u, &S.h[A * 512]);\n")
    # after the TA (b <= A) loop
    rep("        #pragma unroll\n        for (int b = A; b < 4; ++b) {\n            f4 tq = ",
        "        if (nx_) s4bp_ld(io.rdh, io.oh + k2_ * 4096u, &S.dh[A * 512]);\n"
        "        #pragma unroll\n        for (int b = A; b < 4; ++b) {\n            f4 tq = ")
    # before the final math loop
    rep("        const f4 dqi = dqiE, dki = dkp, dgi = dgK;",
        "        if (nx_) s4bp_ld(io.rg, io.og + k2_ * 64u, &S.g[A * 256]);\n"
        "        const f4 dqi = dqiE, dki = dkp, dgi = dgK;")
    # after the final math loop, before the stores
    rep("        dbacc += (dbp[0] + dbp[1]) + (dbp[2] + dbp[3]);\n",
        "        dbacc += (dbp[0] + dbp[1]) + (dbp[2] + dbp[3]);\n"
        "        if (nx_) {\n"
        "            if constexpr (A < 2) {\n"
        "                s4bp_ld(io.rdw, io.odw + k2_ * 2048u, &S.dwT[A * 512]);\n"
        "            } else {\n"
        "                const unsigned o_ = io.okq == kS4bpOOB ? kS4bpOOB : io.okq + k2_ * 32u;\n"
        "                s4bp_ld(io.rk, o_, &S.k[(A - 2) * 512]);\n"
        "                s4bp_ld(io.rq, o_, &S.q[(A - 2) * 512]);\n"
        "            }\n"
        "        }\n")

if "h3" in want:
    rep("            #pragma unroll\n            for (int e = 0; e < 4; ++e) {\n                const float c = s4bp_msum(kd[e]);\n"
        "                if (m == 0) sm.cs2[buf][A][4 * q + e] = c;\n            }",
        "            #pragma unroll\n            for (int e = 0; e < 4; ++e) {   // kra P-37h3: all 16 lanes hold the same sum\n"
        "                const float c = s4bp_msum(kd[e]);\n                sm.cs2[buf][A][4 * q + e] = c;\n            }")
    rep("            hd = xsum16(hd);\n            if (q == 0) sm.cs1[buf][A][pm] = hd;",
        "            hd = xsum16(hd);\n            sm.cs1[buf][A][pm] = hd;                      // kra P-37h3: 4 lanes, same value")
    rep("        if (m == 8) {\n            #pragma unroll\n            for (int e = 0; e < 4; ++e) sm.gM[buf][A][4 * e + q] = gv[e];\n        }",
        "        #pragma unroll                                  // kra P-37h3: g8 == the m==8 lane's gv\n"
        "        for (int e = 0; e < 4; ++e) sm.gM[buf][A][4 * e + q] = g8[e];")
    rep("""            if (row == rlast) {
                const int c = 4 * q + e;
                const float s1 = ((sm.cs1[buf][0][c] + sm.cs1[buf][1][c]) + sm.cs1[buf][2][c]) + sm.cs1[buf][3][c];
                const float s2 = ((sm.cs2[buf][0][c] + sm.cs2[buf][1][c]) + sm.cs2[buf][2][c]) + sm.cs2[buf][3][c];
                dg += s1 * ex2(gn[e]) + s2;
            }""", """            {                                           // kra P-37h3: select, never "+ 0"
                const int c = 4 * q + e;
                const float s1 = ((sm.cs1[buf][0][c] + sm.cs1[buf][1][c]) + sm.cs1[buf][2][c]) + sm.cs1[buf][3][c];
                const float s2 = ((sm.cs2[buf][0][c] + sm.cs2[buf][1][c]) + sm.cs2[buf][2][c]) + sm.cs2[buf][3][c];
                const float dgc = dg + (s1 * ex2(gn[e]) + s2);
                dg = row == rlast ? dgc : dg;
            }""")
if "h5" in want:   # uniform loop: no kt-dependent branch left inside the kt loop
    rep("""        if (kt == 0) s4bp_vmcnt<0>();                    // kra P-37e: stage 1 issued after the prologue
        else if (kt + 1 < 8) s4bp_vmcnt<3>();""",
        """        s4bp_vmcnt<3>();          // kra P-37h5: stage kt+1 (stage 1: drained before the loop)""")
    rep("        if (kt + 2 < 8) s4bp_issue<A>(io, S, kt + 2);      // tile kt+2 -> the stage kt freed\n",
        "        s4bp_issue_or_oob<A>(io, S, kt + 2);               // kra P-37h5: OOB (zeros) past tile 7\n")
    rep("    float dbacc = 0.f;\n", "    s4bp_vmcnt<0>();                                       // kra P-37h5: stage 1 landed\n    float dbacc = 0.f;\n")
    # the OOB issue of tile 9 is never consumed: drain it before the wave exits, or its LDS-DMA
    # zeros could land in the LDS of the next CTA scheduled on this CU
    rep("    db = xsum16(db) + xsum16(dbacc);\n", "    s4bp_vmcnt<0>();                                       // kra P-37h5: no LDS-DMA in flight at exit\n    db = xsum16(db) + xsum16(dbacc);\n")
    helper = """
// kra P-37h5: like s4bp_issue, but for tiles past the end every lane gets an out-of-range offset
// (the hardware deposits zeros into the already-consumed stage) -> the op count per kt is uniform
template <int A>
__device__ __forceinline__ void s4bp_issue_or_oob(const S4BPIO& io, S4BPStage& S, int kt) {
    const bool v = kt < 8;
    const unsigned o16 = v ? io.oh + (unsigned)kt * 4096u : kS4bpOOB;
    s4bp_ld(io.rh, o16, &S.h[A * 512]);
    s4bp_ld(io.rdh, o16, &S.dh[A * 512]);
    s4bp_ld(io.rg, v ? io.og + (unsigned)kt * 64u : kS4bpOOB, &S.g[A * 256]);
    if constexpr (A < 2) {
        s4bp_ld(io.rdw, v ? io.odw + (unsigned)kt * 2048u : kS4bpOOB, &S.dwT[A * 512]);
    } else {
        const unsigned o = (!v || io.okq == kS4bpOOB) ? kS4bpOOB : io.okq + (unsigned)kt * 32u;
        s4bp_ld(io.rk, o, &S.k[(A - 2) * 512]);
        s4bp_ld(io.rq, o, &S.q[(A - 2) * 512]);
    }
}
"""
    anc = "template <int A, int M>\n__device__ __forceinline__ void s4bp_tail("
    assert seg.count(anc) == 1
    seg = seg.replace(anc, helper + "\n" + anc)

if "h6" in want:  # sched_group_barrier: interleave LDS reads, MMAC and VALU in the front region
    rep("        s4bp_vmcnt<3>();          // kra P-37h5: stage kt+1 (stage 1: drained before the loop)",
        """        // kra P-37h6: front region order -- 4 LDS reads up front, then 20 x {1 LDS read, 1 MMAC, 4 VALU}
        __builtin_amdgcn_sched_group_barrier(0x100, 4, 0);
        #pragma unroll
        for (int z_ = 0; z_ < 20; ++z_) {
            __builtin_amdgcn_sched_group_barrier(0x100, 1, 0);
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
            __builtin_amdgcn_sched_group_barrier(0x002, 4, 0);
        }
        s4bp_vmcnt<3>();          // kra P-37h5: stage kt+1 (stage 1: drained before the loop)""")

t = t[:a] + seg + t[b:]
P.write_text(t)
print("P-37h applied:", sorted(want))
