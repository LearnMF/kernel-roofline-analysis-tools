"""kra P-37b (round 12, s4b rewrite step 2, A-class = bit-identical): cross-iteration software
pipeline inside the P-37 pipelined s4b, to give the single wave per SIMD independent work.

P-37's SQTT showed VMEM waits gone but the wave's own dependency chains exposed (LDS -> MMAC,
ds_bpermute reductions, two barriers per kt).  Each kt is now split into
  F(t)  "front": reads stage t (h/dh/k/q/g/dwT), pass-K and pass-Q MMAC chains, kd/msum, hd/xsum16,
        the dwT epilogue product, ep/en -- ONLY LDS reads, results kept in registers;
  T(t)  "tables": writes cs1/cs2[t&1], TA/TQ/TK, gM of tile t (after the WAR barrier);
  B(t)  "back": the intra-chunk table products, the final math and the three stores of tile t.
Iteration kt runs F(kt+1) and B(kt) in ONE basic block with no LDS store between them (two
independent chains the scheduler can interleave), then the WAR barrier, then T(kt+1):
    pre-loop: F(0); T(0)
    kt = 0..6:  [wait stage kt+1] barrier(M) -> issue stage kt+2 -> F(kt+1) || B(kt) -> stores(kt)
                barrier(W) -> T(kt+1)
    kt = 7:     barrier(M) -> B(7) -> stores(7)
Stage kt+1 is waited at M_kt: after its group this wave issued only stores(kt-1) -> vmcnt(3).
Every floating-point expression is unchanged from s4b_tail / P-37 (the rlast correction becomes a
select of the corrected value, never an added zero, so -0.0 survives).
Apply AFTER p37_s4b_pipe.py.   Usage: python3 p37b_s4b_pipe_ilp.py <tree root>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wy.cuh"
t = P.read_text()
head = "    float dbacc = 0.f;\n    #pragma unroll 1\n    for (int kt = 0; kt < 8; ++kt) {\n        const int buf = kt & 1;\n        S4BPStage& S = sm.st[buf];"
tail = "    db = xsum16(db) + xsum16(dbacc);\n    if (q == 0 && rok) p.db[(size_t)(tok0 + row) * H + hh] = db;\n}\n\ntemplate <bool VL>\n__global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(1, 1)))\np3_wy_s4b_pipe_kernel"
i = t.index(head)
j = t.index(tail, i)
assert t.count(head) == 1

NEW = r'''    float dbacc = 0.f;
    // ---- P-37b: F(t) front (LDS reads only, results in registers) ----
    struct Front { v4bh kk, qq; f4 gv, gn, dkp, dgK, dqiE, acc, ep, en; float c2[4]; float hd; };
    auto front = [&](int t_, Front& F) {
        const S4BPStage& S = sm.st[t_ & 1];
        const bf16* sh = &S.h[0];
        const bf16* sdh = &S.dh[0];
        auto HB = [&](int vt) { return ld8(sh + vt * 256 + l * 4); };
        auto DHB = [&](int vt) { return ld8(sdh + vt * 256 + l * 4); };
        F.gv = *reinterpret_cast<const f4*>(&S.g[row * 16 + 4 * q]);
        F.kk = ld8(&S.k[row * 16 + 4 * q]);
        F.qq = ld8(&S.q[row * 16 + 4 * q]);
        F.gn = *reinterpret_cast<const f4*>(&S.g[(kBT - 1) * 16 + 4 * q]);
        {   // pass K tile t  [verbatim]
            #pragma clang fp contract(off)
            f4 a4[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) a4[j] = f4{0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) a4[vt & 3] = mmac(avn[vt], DHB(vt), a4[vt & 3]);
            const f4 acck = (a4[0] + a4[1]) + (a4[2] + a4[3]);
            float kd[4];
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                F.dkp[e] = acck[e] * ex2(F.gn[e] - F.gv[e]);
                kd[e] = bf2f(F.kk[e]) * F.dkp[e];
                F.dgK[e] = -kd[e];
            }
            #pragma unroll
            for (int e = 0; e < 4; ++e) F.c2[e] = msum(kd[e]);
        }
        {
            float hd = 0.f;
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const v4bh h0 = HB(2 * A + j), d0 = DHB(2 * A + j);
                #pragma unroll
                for (int e = 0; e < 4; ++e) hd += bf2f(h0[e]) * bf2f(d0[e]);
            }
            F.hd = xsum16(hd);
        }
        {
            #pragma clang fp contract(off)
            f4 a4[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) a4[j] = f4{0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) a4[vt & 3] = mmac(ado[vt], HB(vt), a4[vt & 3]);
            const f4 accq = (a4[0] + a4[1]) + (a4[2] + a4[3]);
            #pragma unroll
            for (int e = 0; e < 4; ++e) F.dqiE[e] = accq[e] * ex2(F.gv[e]) * p.scale;
        }
        F.acc = f4{0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int st = 0; st < 4; ++st) {
            const int c4 = 4 * st + q;
            F.acc = mmac(aa[st], ld8(&S.dwT[(pm * 8 + ((c4 >> 1) ^ (pm & 7))) * 8 + 4 * (c4 & 1)]), F.acc);
        }
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float d = F.gv[e] - __shfl(F.gv[e], 16 * q + 8, 64);
            F.ep[e] = ex2(d);
            F.en[e] = ex2(-d);
        }
    };
    // ---- T(t): the tile's LDS tables (after the WAR barrier) ----
    auto tables = [&](int t_, const Front& F) {
        const int b_ = t_ & 1;
        if (m == 0) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) sm.cs2[b_][A][4 * q + e] = F.c2[e];
        }
        if (q == 0) sm.cs1[b_][A][pm] = F.hd;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float kf = bf2f(F.kk[e]), qf = bf2f(F.qq[e]);
            const int o = (4 * e + q) * TS + row;
            sm.TA[o] = kf * F.en[e];
            sm.TQ[o] = qf * F.ep[e];
            sm.TK[o] = kf * beta_t * F.ep[e];
        }
        if (m == 8) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) sm.gM[A][4 * e + q] = F.gv[e];
        }
    };
    // ---- B(t): intra-chunk products + final math (reads the tables of tile t) ----
    auto back = [&](int t_, const Front& F, f4& oq, f4& ok, f4& og) {
        const int b_ = t_ & 1;
        f4 cq0 = {0.f, 0.f, 0.f, 0.f}, cq1 = cq0, ck0 = cq0, ck1 = cq0, ct0 = cq0, ct1 = cq0;
        {
        const float gm_n = sm.gM[A][m];
        #pragma unroll
        for (int b = 0; b <= A; ++b) {
            f4 tt = *reinterpret_cast<const f4*>(&sm.TA[m * TS + 16 * b + 4 * q]);
            if (b < A) tt *= ex2_diff(gm_n, sm.gM[b][m]);
            cq0 = mm8(lo2(aqA[b]), lo2(tt), cq0);
            ck0 = mm8(lo2(dak[b]), lo2(tt), ck0);
            cq1 = mm8(hi2(aqA[b]), hi2(tt), cq1);
            ck1 = mm8(hi2(dak[b]), hi2(tt), ck1);
        }
        #pragma unroll
        for (int b = A; b < 4; ++b) {
            f4 tq = *reinterpret_cast<const f4*>(&sm.TQ[m * TS + 16 * b + 4 * q]);
            f4 tk = *reinterpret_cast<const f4*>(&sm.TK[m * TS + 16 * b + 4 * q]);
            if (b > A) {
                const float s = ex2_diff(sm.gM[b][m], gm_n);
                tq *= s;
                tk *= s;
            }
            ct0 = mm8(lo2(aqB[b - A]), lo2(tq), ct0);
            ct1 = mm8(lo2(akB[b - A]), lo2(tk), ct1);
            ct0 = mm8(hi2(aqB[b - A]), hi2(tq), ct0);
            ct1 = mm8(hi2(akB[b - A]), hi2(tk), ct1);
        }
        }
        f4 dbp;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            #pragma clang fp contract(off)
            const float kf = bf2f(F.kk[e]), qf = bf2f(F.qq[e]);
            const float eg = ex2(F.gv[e]);
            const float kg = kf * eg;
            float dk = F.dkp[e] + F.acc[e] * (eg * beta_t);
            float dg = F.dgK[e] + kg * F.acc[e] * beta_t;
            db += F.acc[e] * kg;
            {
                const int c = 4 * q + e;
                const float s1 = ((sm.cs1[b_][0][c] + sm.cs1[b_][1][c]) + sm.cs1[b_][2][c]) + sm.cs1[b_][3][c];
                const float s2 = ((sm.cs2[b_][0][c] + sm.cs2[b_][1][c]) + sm.cs2[b_][2][c]) + sm.cs2[b_][3][c];
                const float dgc = dg + (s1 * ex2(F.gn[e]) + s2);
                dg = row == rlast ? dgc : dg;            // select (never "+ 0"): bits unchanged
            }
            const float dq2 = (cq0[e] + cq1[e]) * F.ep[e];
            const float dk2 = (ck0[e] + ck1[e]) * F.ep[e];
            const float dkt = (ct0[e] + ct1[e]) * F.en[e];
            dbp[e] = dk2 * kf;
            const float dk2b = dk2 * beta_t;
            const float dqf = F.dqiE[e] + dq2;
            oq[e] = dqf;
            ok[e] = dk + dk2b + dkt;
            og[e] = dg + qf * dqf + (dk2b - dkt) * kf;
        }
        dbacc += (dbp[0] + dbp[1]) + (dbp[2] + dbp[3]);
    };
    const unsigned so0 = rok ? (unsigned)(ro * 4) : kS4bpOOB;
    auto store3 = [&](int t_, const f4& oq, const f4& ok, const f4& og) {
        const unsigned so = so0 == kS4bpOOB ? kS4bpOOB : so0 + (unsigned)(t_ * 64);
        s4bp_st3(io.rdq, io.rdk, io.rdg, so, oq, ok, og);
    };

    Front Fc;
    front(0, Fc);
    tables(0, Fc);
    #pragma unroll 1
    for (int kt = 0; kt < 7; ++kt) {
        if (kt > 0) s4bp_vmcnt<3>();                   // stage kt+1 landed (only stores(kt-1) newer)
        lds_barrier();                                 // M: tables(kt) + stage kt+1 visible
        if (kt + 2 < 8) s4bp_issue<A>(io, sm.st[kt & 1], kt + 2);   // stage kt consumed by F(kt)
        Front Fn;
        front(kt + 1, Fn);                             // } one block, two independent chains
        f4 oq, ok, og;
        back(kt, Fc, oq, ok, og);                      // }
        store3(kt, oq, ok, og);
        lds_barrier();                                 // W: tables(kt) reads done
        tables(kt + 1, Fn);
        Fc = Fn;
    }
    lds_barrier();                                     // M(7): tables(7) visible
    {
        f4 oq, ok, og;
        back(7, Fc, oq, ok, og);
        store3(7, oq, ok, og);
    }
'''
t = t[:i] + NEW + t[j:]
P.write_text(t)
print("P-37b (s4b cross-iteration pipeline) APPLIED")
