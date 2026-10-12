"""kra P-37 (round 12, "rewrite s4b" step 1, A-class = must be bit-identical): pipelined s4b.

Why: the G2 s4b hides HBM latency by occupancy (2 CTAs/CU); at 1 CTA/CU it is 0.69-0.82x
(probe_s4b_1cta.py), which blocks every register-hungry fusion (F2/F4).  This kernel hides the
latency by a software pipeline instead:
  * one CTA per CU (51,200 B LDS), two stages of every per-kt operand -- h, dh (4 KB each), g fp32
    (4 KB), dwT (2 KB), k, q (2 KB each) -- staged global->LDS with gfx936
    `buffer_load_dwordx4 ... lds` (no VGPR round trip; masked rows = out-of-range offset -> zeros,
    probe_buffer_lds2.hip T1);
  * the fp32 dq/dk/dg leave through `buffer_store_dwordx4` with out-of-range masking (T2), so the kt
    loop contains NO compiler VMEM op and the hand-placed `s_waitcnt vmcnt(N)` counts are exact;
  * the dwT epilogue product moves before the mid barrier (independent of the tables), so stage kt
    is free after that barrier: tile kt+2 is issued there, and ONE end-of-iteration barrier both
    retires the tables (WAR) and publishes stage kt+1 -- two barriers per kt, as before.
Values: every MMAC/VALU operand and every floating-point expression is copied from s4b_tail; only
where the bytes come from changes.  Selected with KDA_G2_S4B_PIPE=1 (default 0 until measured).
Usage: python3 p37_s4b_pipe.py <tree root>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wy.cuh"
t = P.read_text()

KERNEL = r'''
// ================================================================ kra P-37: pipelined s4b
// (see examples/kda/p37_s4b_pipe.py in kernel-roofline-analysis-tools for the design notes)
typedef int g2_v4i __attribute__((ext_vector_type(4)));
constexpr unsigned kS4bpOOB = 0xFFFFFFF0u;                  // out-of-range buffer offset
__device__ __forceinline__ g2_v4i s4bp_srd(const void* base, unsigned bytes) {
    const unsigned long long a = reinterpret_cast<unsigned long long>(base);
    g2_v4i r;
    r.x = __builtin_amdgcn_readfirstlane((int)(unsigned)a);
    r.y = __builtin_amdgcn_readfirstlane((int)(unsigned)(a >> 32));
    r.z = (int)bytes;
    r.w = (int)0x00020000;
    return r;
}
// each lane's 16 B land at (wave LDS base) + 16 * lane
__device__ __forceinline__ void s4bp_ld(g2_v4i srd, unsigned voff, const void* lds_wave_base) {
    const unsigned m0 = __builtin_amdgcn_readfirstlane((unsigned)(unsigned long long)lds_wave_base);
    asm volatile("s_mov_b32 m0, %0\n\tbuffer_load_dwordx4 %1, %2, 0 offen offset:0 lds"
                 :: "s"(m0), "v"(voff), "s"(srd) : "memory", "m0");
}
__device__ __forceinline__ void s4bp_st(g2_v4i srd, unsigned voff, f4 v) {
    asm volatile("buffer_store_dwordx4 %0, %1, %2, 0 offen offset:0" :: "v"(v), "v"(voff), "s"(srd) : "memory");
}
// vmcnt waits go through the builtin (NOT inline asm) so the compiler's waitcnt pass sees them:
// with an asm wait it still believed the prologue loads were outstanding and re-inserted
// vmcnt(7..0) before every pass-K MMAC inside the loop, draining the async stages each kt.
// gfx9 simm16: vmcnt[3:0] | expcnt[6:4]=7 | lgkmcnt[11:8]=15 | vmcnt[5:4] at [15:14]
template <int N>
__device__ __forceinline__ void s4bp_vmcnt() {
    static_assert(N >= 0 && N < 64, "vmcnt range");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_waitcnt((N & 15) | (((N >> 4) & 3) << 14) | (7 << 4) | (15 << 8));
    __builtin_amdgcn_sched_barrier(0);
}

struct S4BPStage {
    bf16 h[8 * 256];      // tile kt of h, thread order (as stg[buf][0])
    bf16 dh[8 * 256];     // tile kt of dh (as stg[buf][1])
    float g[64 * 16];     // g[row_g(r)][kc..kc+15], r = 0..63 (row 63 = the chunk-end row)
    bf16 dwT[16 * 64];    // dwT rows kc..kc+15; 16 B unit (r, P) at r*8 + (P ^ (r & 7))
    bf16 k[64 * 16];      // k[r][kc..kc+15], zero rows past the chunk end
    bf16 q[64 * 16];
};
struct S4BPSmem {
    float TA[16 * ipi::TS], TQ[16 * ipi::TS], TK[16 * ipi::TS];
    float gM[4][16];
    float cs1[2][4][16], cs2[2][4][16];
    S4BPStage st[2];
};
static_assert(sizeof(S4BPSmem) <= 64 * 1024, "S4BPSmem exceeds the CU's LDS");

struct S4BPIO {           // per-CTA buffer resources + per-thread byte offsets (tile 0)
    g2_v4i rh, rdh, rg, rdw, rk, rq, rdq, rdk, rdg;
    unsigned oh, og, odw, okq;
};

// stage issue for tile kt: waves 0-1 issue h, dh, g, dwT (4 ops); waves 2-3 h, dh, g, k, q (5 ops)
template <int A>
__device__ __forceinline__ void s4bp_issue(const S4BPIO& io, S4BPStage& S, int kt) {
    s4bp_ld(io.rh, io.oh + (unsigned)kt * 4096u, &S.h[A * 512]);
    s4bp_ld(io.rdh, io.oh + (unsigned)kt * 4096u, &S.dh[A * 512]);
    s4bp_ld(io.rg, io.og == kS4bpOOB ? kS4bpOOB : io.og + (unsigned)kt * 64u, &S.g[A * 256]);
    if constexpr (A < 2) {
        s4bp_ld(io.rdw, io.odw + (unsigned)kt * 2048u, &S.dwT[A * 512]);
    } else {
        const unsigned o = io.okq == kS4bpOOB ? kS4bpOOB : io.okq + (unsigned)kt * 32u;
        s4bp_ld(io.rk, o, &S.k[(A - 2) * 512]);
        s4bp_ld(io.rq, o, &S.q[(A - 2) * 512]);
    }
}

template <int A, int M>
__device__ __forceinline__ void s4bp_tail(const WyS4BParams& p, S4BPSmem& sm, const v4bh (&aa)[4],
                                          float db, float beta_t, long tok0, int cl) {
    using namespace ipi;
    constexpr int NA = A < 2 ? 4 : 5;                       // async ops per stage of this wave
    const int l = threadIdx.x & 63, m = l & 15, q = l >> 4, pm = pi16(m);
    const int tid = threadIdx.x;
    const int i = blockIdx.x, hh = blockIdx.y, H = p.H;
    const int R = H * kD, RA = H * kBT;
    const size_t cb = g2_rm_tok(tok0, hh, H, kD), cbA = g2_rm_tok(tok0, hh, H, kBT);
    const size_t cbAp = g2_rm_tok((long)i * kBT, hh, H, kBT);   // dA handoff (chunk-padded)
    const int row = 16 * A + m;
    const bool rok = g2_row_ok<M>(row, cl);
    const int ro = row * R + 4 * q;
    const int rlast = M == 2 ? cl - 1 : kBT - 1;                    // the chunk-end dg term's row
    const float* aq_ = p.dAqk + cbA;
    const float* dAg = p.dA + cbAp;
    const size_t cQ = g2_nat_blk(i, hh, H, 0, true, 8 * 8 * 256);

    // ---- async staging set-up and the first two stages (issued before the prologue loads)
    S4BPIO io;
    io.rh = s4bp_srd(p.h + cQ, 8 * 8 * 256 * 2);
    io.rdh = s4bp_srd(p.dh + cQ, 8 * 8 * 256 * 2);
    io.rg = s4bp_srd(p.g + cb, (unsigned)(kBT * R * 4));
    io.rdw = s4bp_srd(p.dwT + g2_nat_blk(i, hh, H, 0, true, kD * kBT), kD * kBT * 2);
    io.rk = s4bp_srd(p.k + cb, (unsigned)(kBT * R * 2));
    io.rq = s4bp_srd(p.q_row + cb, (unsigned)(kBT * R * 2));
    io.rdq = s4bp_srd(p.dq + cb, (unsigned)(kBT * R * 4));
    io.rdk = s4bp_srd(p.dk + cb, (unsigned)(kBT * R * 4));
    io.rdg = s4bp_srd(p.dg + cb, (unsigned)(kBT * R * 4));
    io.oh = (unsigned)tid * 16u;
    {
        const int gr = tid >> 2, gp = tid & 3;                      // g: 4 lanes per row
        io.og = (unsigned)((g2_row_g<M>(gr, cl) * R + 4 * gp) * 4);
    }
    {
        const int u = tid & 127, r_ = u >> 3, P_ = (u & 7) ^ (r_ & 7);   // dwT: inverse swizzle
        io.odw = (unsigned)((r_ * kBT + 8 * P_) * 2);
        const int kr = u >> 1, kh = u & 1;                          // k/q: 2 lanes per row
        io.okq = g2_row_ok<M>(kr, cl) ? (unsigned)((kr * R + 8 * kh) * 2) : kS4bpOOB;
    }
    s4bp_issue<A>(io, sm.st[0], 0);
    s4bp_issue<A>(io, sm.st[1], 1);

    f4 dak[4];
    #pragma unroll
    for (int ut = 0; ut < 4; ++ut) dak[ut] = ld16f(dAg + row * RA + 16 * ut + 4 * q);
    f4 aqA[A + 1], aqB[4 - A], akB[4 - A];
    #pragma unroll
    for (int b = 0; b <= A; ++b) {
        aqA[b] = rok ? ld16f(aq_ + row * RA + 16 * b + 4 * q) : f4{0.f, 0.f, 0.f, 0.f};
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
            akB[b - A][x] = keep ? dAg[j * RA + row] : 0.f;   // dAT[row][j] == dA[j][row]
            aqB[b - A][x] = (keep && g2_row_ok<M>(j, cl)) ? aq_[j * RA + row] : 0.f;
        }
    v4bh ado[8], avn[8];
    {
        if (p.do_ != nullptr) {
            const bf16* doN = p.do_ + g2_nat_blk(i, hh, H, 0, true, 4 * 8 * 256);
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) ado[vt] = ld8(doN + ((A * 8 + vt) * 64 + l) * 4);
        } else {
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt)
                ado[vt] = rok ? ld8(p.do_row + cb + ro + 16 * vt) : v4bh{0, 0, 0, 0};
        }
        if (p.v_new != nullptr) {
            const bf16* vnN = p.v_new + g2_nat_blk(i, hh, H, 0, true, 4 * 8 * 256);
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) avn[vt] = ld8(vnN + ((A * 8 + vt) * 64 + l) * 4);
        } else {
            const bf16* vnc = p.vnc + g2_nat_blk(i, hh, H, p.NT, false, 8192) + A * 8 * 256 + l * 4;
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) avn[vt] = g2_c2a(ld8(vnc + vt * 256), l);
        }
    }
    // Retire every loop-invariant global load HERE, visibly to the compiler: otherwise its waitcnt
    // dataflow merges a "pending" state into the loop header and re-waits vmcnt(7..0) every kt,
    // which would also drain the async stages.
    #pragma unroll
    for (int vt = 0; vt < 8; ++vt) asm volatile("" :: "v"(avn[vt]), "v"(ado[vt]));
    #pragma unroll
    for (int b = 0; b < 4; ++b) asm volatile("" :: "v"(dak[b]));
    #pragma unroll
    for (int b = 0; b <= A; ++b) asm volatile("" :: "v"(aqA[b]));
    #pragma unroll
    for (int b = A; b < 4; ++b) asm volatile("" :: "v"(aqB[b - A]), "v"(akB[b - A]));
    asm volatile("" :: "v"(db), "v"(beta_t));
    s4bp_vmcnt<0>();
    lds_barrier();                                         // stage 0 (and 1) visible

    float dbacc = 0.f;
    #pragma unroll 1
    for (int kt = 0; kt < 8; ++kt) {
        const int buf = kt & 1;
        S4BPStage& S = sm.st[buf];
        const bf16* sh = &S.h[0];
        const bf16* sdh = &S.dh[0];
        auto HB = [&](int vt) { return ld8(sh + vt * 256 + l * 4); };
        auto DHB = [&](int vt) { return ld8(sdh + vt * 256 + l * 4); };
        const f4 gv = *reinterpret_cast<const f4*>(&S.g[row * 16 + 4 * q]);
        const v4bh kk = ld8(&S.k[row * 16 + 4 * q]);
        const v4bh qq = ld8(&S.q[row * 16 + 4 * q]);
        const f4 gn = *reinterpret_cast<const f4*>(&S.g[(kBT - 1) * 16 + 4 * q]);
        f4 dkp, dgK;
        {   // pass K tile kt  [verbatim]
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
                dkp[e] = acck[e] * ex2(gn[e] - gv[e]);
                kd[e] = bf2f(kk[e]) * dkp[e];
                dgK[e] = -kd[e];
            }
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float c = msum(kd[e]);
                if (m == 0) sm.cs2[buf][A][4 * q + e] = c;
            }
        }
        {
            float hd = 0.f;
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const v4bh h0 = HB(2 * A + j), d0 = DHB(2 * A + j);
                #pragma unroll
                for (int e = 0; e < 4; ++e) hd += bf2f(h0[e]) * bf2f(d0[e]);
            }
            hd = xsum16(hd);
            if (q == 0) sm.cs1[buf][A][pm] = hd;
        }
        f4 dqiE;
        {
            #pragma clang fp contract(off)
            f4 a4[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) a4[j] = f4{0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) a4[vt & 3] = mmac(ado[vt], HB(vt), a4[vt & 3]);
            const f4 accq = (a4[0] + a4[1]) + (a4[2] + a4[3]);
            #pragma unroll
            for (int e = 0; e < 4; ++e) dqiE[e] = accq[e] * ex2(gv[e]) * p.scale;
        }
        // p3 epilogue product dkgb = A^T (-dw), tile kt (moved before the mid barrier: it reads
        // only aa + the staged dwT, so stage kt is free once that barrier passes)
        f4 acc = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int st = 0; st < 4; ++st) {
            const int c4 = 4 * st + q;
            acc = mmac(aa[st], ld8(&S.dwT[(pm * 8 + ((c4 >> 1) ^ (pm & 7))) * 8 + 4 * (c4 & 1)]), acc);
        }
        f4 ep, en;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float d = gv[e] - __shfl(gv[e], 16 * q + 8, 64);
            ep[e] = ex2(d);
            en[e] = ex2(-d);
        }
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float kf = bf2f(kk[e]), qf = bf2f(qq[e]);
            const int o = (4 * e + q) * TS + row;
            sm.TA[o] = kf * en[e];
            sm.TQ[o] = qf * ep[e];
            sm.TK[o] = kf * beta_t * ep[e];
        }
        if (m == 8) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) sm.gM[A][4 * e + q] = gv[e];
        }
        lds_barrier();                                     // tables visible; stage kt consumed
        if (kt + 2 < 8) s4bp_issue<A>(io, S, kt + 2);      // tile kt+2 -> the stage kt freed

        f4 cq0 = {0.f, 0.f, 0.f, 0.f}, cq1 = cq0, ck0 = cq0, ck1 = cq0, ct0 = cq0, ct1 = cq0;
        {
        const float gm_n = sm.gM[A][m];
        #pragma unroll
        for (int b = 0; b <= A; ++b) {
            f4 t = *reinterpret_cast<const f4*>(&sm.TA[m * TS + 16 * b + 4 * q]);
            if (b < A) t *= ex2_diff(gm_n, sm.gM[b][m]);
            cq0 = mm8(lo2(aqA[b]), lo2(t), cq0);
            ck0 = mm8(lo2(dak[b]), lo2(t), ck0);
            cq1 = mm8(hi2(aqA[b]), hi2(t), cq1);
            ck1 = mm8(hi2(dak[b]), hi2(t), ck1);
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
        const f4 dqi = dqiE, dki = dkp, dgi = dgK;
        f4 oq, ok, og, dbp;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            #pragma clang fp contract(off)
            const float kf = bf2f(kk[e]), qf = bf2f(qq[e]);
            const float eg = ex2(gv[e]);
            const float kg = kf * eg;
            float dk = dki[e] + acc[e] * (eg * beta_t);
            float dg = dgi[e] + kg * acc[e] * beta_t;
            db += acc[e] * kg;
            if (row == rlast) {
                const int c = 4 * q + e;
                const float s1 = ((sm.cs1[buf][0][c] + sm.cs1[buf][1][c]) + sm.cs1[buf][2][c]) + sm.cs1[buf][3][c];
                const float s2 = ((sm.cs2[buf][0][c] + sm.cs2[buf][1][c]) + sm.cs2[buf][2][c]) + sm.cs2[buf][3][c];
                dg += s1 * ex2(gn[e]) + s2;
            }
            const float dq2 = (cq0[e] + cq1[e]) * ep[e];
            const float dk2 = (ck0[e] + ck1[e]) * ep[e];
            const float dkt = (ct0[e] + ct1[e]) * en[e];
            dbp[e] = dk2 * kf;
            const float dk2b = dk2 * beta_t;
            const float dqf = dqi[e] + dq2;
            oq[e] = dqf;
            ok[e] = dk + dk2b + dkt;
            og[e] = dg + qf * dqf + (dk2b - dkt) * kf;
        }
        dbacc += (dbp[0] + dbp[1]) + (dbp[2] + dbp[3]);
        const unsigned so = rok ? (unsigned)((ro + 16 * kt) * 4) : kS4bpOOB;
        s4bp_st(io.rdq, so, oq);
        s4bp_st(io.rdk, so, ok);
        s4bp_st(io.rdg, so, og);
        // stage kt+1 must be complete before the barrier publishes it.  Ops this wave issued after
        // stage kt+1's group: [stores kt-1 (3)] + [stage kt+2 (NA) if issued] + [stores kt (3)].
        if (kt == 0) s4bp_vmcnt<3 + NA>();
        else if (kt + 2 < 8) s4bp_vmcnt<6 + NA>();
        else if (kt + 1 < 8) s4bp_vmcnt<6>();
        lds_barrier();                                     // table WAR + stage kt+1 visible
    }
    db = xsum16(db) + xsum16(dbacc);
    if (q == 0 && rok) p.db[(size_t)(tok0 + row) * H + hh] = db;
}

template <bool VL>
__global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(1, 1)))
p3_wy_s4b_pipe_kernel(WyS4BParams p) {
    __shared__ __align__(16) S4BPSmem sm;
    const int tid = threadIdx.x, r = tid >> 6, l = tid & 63, m = l & 15, q = l >> 4;
    const int i = blockIdx.x, hh = blockIdx.y, H = p.H;
    const int tr = 16 * r + m;
    const long tok0 = g2_row_tok<VL>(p.ctok, i);
    const int cl = g2_row_len<VL>(p.clen, i);
    g2_vl_dispatch<VL>(cl, [&](auto mode_) {
        constexpr int M = decltype(mode_)::value;
        const size_t cbA = g2_rm_tok(tok0, hh, H, kBT);
        const int RA = H * kBT;
        const bf16* A_ = p.A + cbA;
        const float* beta_ = p.beta + (size_t)tok0 * H + hh;
        v4bh aa[4];
        {   // Y4 staging of A in stage 0's h+dh (8 KB, contiguous), transient
            short* As = reinterpret_cast<short*>(&sm.st[0].h[0]);
            #pragma unroll
            for (int it_ = 0; it_ < 4; ++it_) {
                const int j = tid + 256 * it_;
                const int row = j >> 4, qc = j & 15;
                *reinterpret_cast<v4bh*>(As + row * 64 + 4 * qc) = g2_row_ok<M>(row, cl)
                    ? *reinterpret_cast<const v4bh*>(A_ + row * RA + 4 * qc) : v4bh{0, 0, 0, 0};
            }
            lds_barrier();
            #pragma unroll
            for (int st = 0; st < 4; ++st)
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    aa[st][e] = g2_row_ok<M>(16 * st + 4 * q + e, cl)
                        ? As[(16 * st + 4 * q + e) * 64 + tr] : (short)0;
            lds_barrier();                                  // aa out; stage 0 reusable
        }
        const float beta_t = g2_row_ok<M>(tr, cl) ? beta_[tr * H] : 0.f;
        const float db0 = p.db_part[g2_nat_blk(i, hh, H, 0, true, 256) + tid];
        switch (__builtin_amdgcn_readfirstlane(r)) {
            case 0: s4bp_tail<0, M>(p, sm, aa, db0, beta_t, tok0, cl); break;
            case 1: s4bp_tail<1, M>(p, sm, aa, db0, beta_t, tok0, cl); break;
            case 2: s4bp_tail<2, M>(p, sm, aa, db0, beta_t, tok0, cl); break;
            default: s4bp_tail<3, M>(p, sm, aa, db0, beta_t, tok0, cl); break;
        }
    });
}
'''

anchor = 'extern "C" int g2_wy_s4_split_r('
assert t.count(anchor) == 1, t.count(anchor)
t = t.replace(anchor, KERNEL + "\n" + anchor)

old = """        if (ctok) hipLaunchKernelGGL(p3_wy_s4b_kernel<true>, dim3(NT, H), dim3(256), 0, s, pb);
        else hipLaunchKernelGGL(p3_wy_s4b_kernel<false>, dim3(NT, H), dim3(256), 0, s, pb);"""
new = """        static const bool pipe = [] { const char* e = getenv("KDA_G2_S4B_PIPE"); return e && atoi(e) == 1; }();
        if (pipe) {   // kra P-37: pipelined s4b (1 CTA/CU, async LDS staging)
            if (ctok) hipLaunchKernelGGL(p3_wy_s4b_pipe_kernel<true>, dim3(NT, H), dim3(256), 0, s, pb);
            else hipLaunchKernelGGL(p3_wy_s4b_pipe_kernel<false>, dim3(NT, H), dim3(256), 0, s, pb);
        } else if (ctok) hipLaunchKernelGGL(p3_wy_s4b_kernel<true>, dim3(NT, H), dim3(256), 0, s, pb);
        else hipLaunchKernelGGL(p3_wy_s4b_kernel<false>, dim3(NT, H), dim3(256), 0, s, pb);"""
assert t.count(old) == 1
t = t.replace(old, new)
P.write_text(t)
print("P-37 (pipelined s4b) APPLIED")
