"""kra P-39 = fusion F6 (round 12): s4a and s4b in ONE kernel (one CTA per (chunk, head), 1 CTA/CU).

What disappears (per chunk-head): the dwT (bf16 128x64) and dA (fp32 64x64) global write + read,
the db_part handoff, the second Y4 staging of A, a launch, and the second HBM read of the h block
(phase B re-reads it shortly after phase A -> mostly L2).  At 8K/96 about 0.8 GB of round trips.
Values are NOT re-rounded: dwT stays bf16 in LDS (exactly the bits s4a copied to global), dA stays
fp32 (registers for the lane's own rows, LDS for the transposed gathers), db stays the register s4a
stored -> bit-identical to the split pair (gate: 19-config bitwise).

LDS (64,512 B):  S [128][68] bf16 (phase A dvT->dwT, phase B reads dwT) | union{ phase A: XT, KGs,
BETAs ; phase B: single-buffered TA/TQ/TK, gM, cs1/cs2 } | 2 stages x 16 KB {h, dh, g, k, q}.
The prologue tiles for phase B ride in the stages: dAqk -> stage 0 (async, issued at kernel start),
dA -> stage 1 (written by phase A's dak loop).  Phase B = the P-37 loop (two barriers per kt) with
P-37c (DPP msum, ep/en partner from the g tile), dwT read straight from S, 4 async ops per wave
per stage (h, dh, g + k for waves 0-1 / q for waves 2-3).
Requires p37_s4b_pipe + p37c + p37f applied (helpers).  Selected with KDA_G2_S4AB=1.
Usage: python3 p39_f6_s4ab.py <tree>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_wy.cuh"
t = P.read_text()
assert "s4bp_msum" in t and "s4bp_c2a" in t, "apply p37 + p37c + p37f first"

KERNEL = r'''
// ================================================================ kra P-39 (F6): s4a + s4b fused
struct S4ABStage {
    bf16 h[8 * 256];      // tile kt of h (thread order)
    bf16 dh[8 * 256];     // tile kt of dh
    float g[64 * 16];     // g[row_g(r)][kc..kc+15]
    bf16 k[64 * 16];      // zero rows past the chunk end
    bf16 q[64 * 16];
};
struct S4ABSmem {
    bf16 S[kD * kPT];                                   // phase A dvT -> dwT; phase B reads dwT
    union {
        struct { bf16 XT[kBT * kPT]; bf16 KGs[2][4 * 64 * 4]; float BETAs[kBT]; } a;
        struct { float TA[16 * ipi::TS], TQ[16 * ipi::TS], TK[16 * ipi::TS];
                 float gM[4][16]; float cs1[2][4][16], cs2[2][4][16]; } b;
    } u;
    S4ABStage st[2];
};
static_assert(sizeof(S4ABSmem) <= 64 * 1024, "S4ABSmem exceeds the CU's LDS");

template <int A>
__device__ __forceinline__ void s4ab_issue(const S4BPIO& io, S4ABStage& S, int kt) {
    s4bp_ld(io.rh, io.oh + (unsigned)kt * 4096u, &S.h[A * 512]);
    s4bp_ld(io.rdh, io.oh + (unsigned)kt * 4096u, &S.dh[A * 512]);
    s4bp_ld(io.rg, io.og + (unsigned)kt * 64u, &S.g[A * 256]);
    const unsigned o = io.okq == kS4bpOOB ? kS4bpOOB : io.okq + (unsigned)kt * 32u;
    if constexpr (A < 2) s4bp_ld(io.rk, o, &S.k[A * 512]);
    else s4bp_ld(io.rq, o, &S.q[(A - 2) * 512]);
}

template <int A, int M, bool AFS>
__device__ __forceinline__ void s4ab_body(const WyS4AParams& pa, const WyS4BParams& p, S4ABSmem& sm,
                                          long tok0, int cl) {
    using namespace ipi;
    constexpr int NA = 4;
    const int tid = threadIdx.x, l = tid & 63, m = l & 15, q = l >> 4, pm = pi16(m);
    const int i = blockIdx.x, hh = blockIdx.y, H = p.H;
    const int tr = 16 * A + m, row = tr;
    const bool trok = g2_row_ok<M>(tr, cl), rok = trok;
    const size_t cb = g2_rm_tok(tok0, hh, H, kD), cbA = g2_rm_tok(tok0, hh, H, kBT);
    const int R = H * kD, RA = H * kBT;
    const int ro = tr * R + 4 * q;
    const int rlast = M == 2 ? cl - 1 : kBT - 1;
    bf16* const S = sm.S;
    float* const lQ = reinterpret_cast<float*>(&sm.st[0]);     // dAqk tile (prologue of phase B)
    float* const lA = reinterpret_cast<float*>(&sm.st[1]);     // dA tile  (written by phase A)

    // ---- phase B's prologue inputs, started first: dAqk tile (async), do / v_new fragments
    {
        const g2_v4i rAq = s4bp_srd(p.dAqk + cbA, (unsigned)(kBT * RA * 4));
        #pragma unroll
        for (int it_ = 0; it_ < 4; ++it_) {
            const int u = it_ * 256 + tid, r_ = u >> 4, c_ = u & 15;
            const unsigned off = (unsigned)((r_ * RA + 4 * c_) * 4);
            s4bp_ld(rAq, g2_row_ok<M>(r_, cl) ? off : kS4bpOOB, lQ + (it_ * 256 + A * 64) * 4);
        }
    }
    v4bh ado[8], vraw[8];
    if (p.do_ != nullptr) {
        const bf16* doN = p.do_ + g2_nat_blk(i, hh, H, 0, true, 4 * 8 * 256);
        #pragma unroll
        for (int vt = 0; vt < 8; ++vt) ado[vt] = ld8(doN + ((A * 8 + vt) * 64 + l) * 4);
    } else {
        #pragma unroll
        for (int vt = 0; vt < 8; ++vt)
            ado[vt] = rok ? ld8(p.do_row + cb + ro + 16 * vt) : v4bh{0, 0, 0, 0};
    }
    const bool vn_native = p.v_new != nullptr;
    if (vn_native) {
        const bf16* vnN = p.v_new + g2_nat_blk(i, hh, H, 0, true, 4 * 8 * 256);
        #pragma unroll
        for (int vt = 0; vt < 8; ++vt) vraw[vt] = ld8(vnN + ((A * 8 + vt) * 64 + l) * 4);
    } else {
        const bf16* vnc = p.vnc + g2_nat_blk(i, hh, H, p.NT, false, 8192) + A * 8 * 256 + l * 4;
        #pragma unroll
        for (int vt = 0; vt < 8; ++vt) vraw[vt] = ld8(vnc + vt * 256);
    }

    // ================= phase A: s4a body (verbatim apart from the handoffs) =================
    const bf16* v_ = pa.v_row + cb;
    const bf16* A_ = pa.A + cbA;
    const float* beta_ = pa.beta + (size_t)tok0 * H + hh;
    bf16* dv2_ = pa.dv2 + cb;
    const size_t cA = g2_nat_blk(i, hh, H, 0, true, 4 * 8 * 256);
    const size_t cB = g2_nat_blk(i, hh, H, 0, true, 8 * 8 * 256);
    const bf16* hb = pa.h + cB;
    const bf16* dvN = pa.dv + cA;
    const bf16* vBN = pa.v_vb + cA;
    const bf16* kgB = pa.kgb + cA;
    bf16* const XT = sm.u.a.XT;
    auto aN = [&](const bf16* base, int vt) { return ld8(base + ((A * 8 + vt) * 64 + l) * 4); };

    const float beta_t = trok ? beta_[tr * H] : 0.f;
    if (tid < kBT) sm.u.a.BETAs[tid] = g2_row_ok<M>(tid, cl) ? beta_[tid * H] : 0.f;
    v4bh aa[4];
    {   // Y4: A block staged in S (transient), transposed column gathered from LDS
        short* As = reinterpret_cast<short*>(S);
        #pragma unroll
        for (int it_ = 0; it_ < 4; ++it_) {
            const int j = tid + 256 * it_;
            const int rr = j >> 4, qc = j & 15;
            *reinterpret_cast<v4bh*>(As + rr * 64 + 4 * qc) = g2_row_ok<M>(rr, cl)
                ? *reinterpret_cast<const v4bh*>(A_ + rr * RA + 4 * qc) : v4bh{0, 0, 0, 0};
        }
        lds_barrier();
        #pragma unroll
        for (int st = 0; st < 4; ++st)
            #pragma unroll
            for (int e = 0; e < 4; ++e)
                aa[st][e] = g2_row_ok<M>(16 * st + 4 * q + e, cl)
                    ? As[(16 * st + 4 * q + e) * 64 + tr] : (short)0;
        lds_barrier();
    }
    v4bh dva[8];
    #pragma unroll
    for (int vt = 0; vt < 8; ++vt) dva[vt] = aN(dvN, vt);
    #pragma unroll
    for (int vt = 0; vt < 8; ++vt)
        #pragma unroll
        for (int e = 0; e < 4; ++e) reinterpret_cast<short*>(S)[(16 * vt + 4 * q + e) * kPT + tr] = dva[vt][e];
    lds_barrier();                                                      // dvT
    float db = 0.f;
    v4bh vrow[8];
    #pragma unroll
    for (int vt = 0; vt < 8; ++vt) {
        f4 acc = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int st = 0; st < 4; ++st) acc = mmac(aa[st], ld8(S + (16 * vt + pm) * kPT + 16 * st + 4 * q), acc);
        const v4bh vv = trok ? ld8(v_ + ro + 16 * vt) : v4bh{0, 0, 0, 0};
        vrow[vt] = vv;
        v4bh o;
        #pragma unroll
        for (int e = 0; e < 4; ++e) { o[e] = bfr(acc[e] * beta_t); db = __builtin_fmaf(acc[e], bf2f(vv[e]), db); }
        if (trok) *reinterpret_cast<v4bh*>(dv2_ + ro + 16 * vt) = o;
    }
    f4 dA[4];
    #pragma unroll
    for (int st = 0; st < 4; ++st) dA[st] = f4{0.f, 0.f, 0.f, 0.f};
    if (pa.v_vb != nullptr) {
        #pragma unroll
        for (int vt = 0; vt < 8; ++vt)
            #pragma unroll
            for (int st = 0; st < 4; ++st)
                dA[st] = mmac(dva[vt], ld8(vBN + ((st * 8 + vt) * 64 + l) * 4), dA[st]);
    } else {
        constexpr int kVS = 2 * kPT;
        lds_barrier();
        #pragma unroll
        for (int vt = 0; vt < 8; ++vt)
            *reinterpret_cast<v4bh*>(reinterpret_cast<short*>(S) + tr * kVS + 16 * vt + 4 * q) = vrow[vt];
        lds_barrier();
        #pragma unroll
        for (int vt = 0; vt < 8; ++vt)
            #pragma unroll
            for (int st = 0; st < 4; ++st)
                dA[st] = mmac(dva[vt], ld8(S + (16 * st + pm) * kVS + 16 * vt + 4 * q), dA[st]);
    }
    lds_barrier();                                                      // dvT/v tile dead -> dwT
    const v4bh (&af)[8] = dva;
    {   // pass W: -dw -> dwT (S), dA += (-dw) kg^T
        auto body = [&](int kt, const v4bh* bf) {
            f4 a4[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) a4[j] = f4{0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) a4[vt & 3] = mmac(af[vt], bf[vt], a4[vt & 3]);
            const f4 acc = (a4[0] + a4[1]) + (a4[2] + a4[3]);
            const int kc = 16 * kt;
            v4bh dwn;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                dwn[e] = bfr(-acc[e]);
                reinterpret_cast<short*>(S)[(kc + 4 * q + e) * kPT + tr] = dwn[e];
            }
            #pragma unroll
            for (int st = 0; st < 4; ++st) {
                const v4bh kgb = ld8(&sm.u.a.KGs[kt & 1][0] + (st * 64 + l) * 4);
                dA[st] = mmac(dwn, kgb, dA[st]);
            }
        };
        const bf16* kgsrc = kgB + (((tid >> 6) * 8) * 64 + (tid & 63)) * 4;
        *reinterpret_cast<v4bh*>(&sm.u.a.KGs[0][0] + 4 * tid) = ld8(kgsrc);
        v4bh kgreg = ld8(kgsrc + 1 * 256);
        short* hs = reinterpret_cast<short*>(XT);
        const bf16* hsrc = hb + 8 * tid;
        f4 hreg = *reinterpret_cast<const f4*>(hsrc);
        *reinterpret_cast<f4*>(hs + 8 * tid) = hreg;
        hreg = *reinterpret_cast<const f4*>(hsrc + 2048);
        lds_barrier();
        #pragma unroll 1
        for (int kt = 0; kt < 8; ++kt) {
            v4bh bf[8];
            const bf16* hk = reinterpret_cast<const bf16*>(hs + (kt & 1) * 2048) + l * 4;
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) bf[vt] = ld8(hk + vt * 256);
            if (kt < 7) {
                *reinterpret_cast<f4*>(hs + ((kt + 1) & 1) * 2048 + 8 * tid) = hreg;
                *reinterpret_cast<v4bh*>(&sm.u.a.KGs[(kt + 1) & 1][0] + 4 * tid) = kgreg;
                if (kt < 6) {
                    hreg = *reinterpret_cast<const f4*>(hsrc + (kt + 2) * 2048);
                    kgreg = ld8(kgsrc + (kt + 2) * 256);
                }
            }
            body(kt, bf);
            lds_barrier();
        }
    }
    v4bh da[4];
    #pragma unroll
    for (int st = 0; st < 4; ++st)
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int s = 16 * st + 4 * q + e;
            da[st][e] = bfr(tr > s ? dA[st][e] * sm.u.a.BETAs[s] : 0.f);
        }
    f4 xc[4];
    if constexpr (AFS) {
        short* As2 = reinterpret_cast<short*>(XT);
        #pragma unroll
        for (int it_ = 0; it_ < 4; ++it_) {
            const int j = tid + 256 * it_;
            const int rr = j >> 4, qc = j & 15;
            *reinterpret_cast<v4bh*>(As2 + rr * kPT + 4 * qc) = g2_row_ok<M>(rr, cl)
                ? *reinterpret_cast<const v4bh*>(A_ + rr * RA + 4 * qc) : v4bh{0, 0, 0, 0};
        }
        lds_barrier();
        #pragma unroll
        for (int ut = 0; ut < 4; ++ut) {
            xc[ut] = f4{0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int st = 0; st < 4; ++st)
                xc[ut] = mmac(da[st], ld8(XT + (16 * ut + pm) * kPT + 16 * st + 4 * q), xc[ut]);
        }
        lds_barrier();
    } else {
        #pragma unroll
        for (int ut = 0; ut < 4; ++ut) {
            xc[ut] = f4{0.f, 0.f, 0.f, 0.f};
            const bool aok = g2_row_ok<M>(16 * ut + pm, cl);
            #pragma unroll
            for (int st = 0; st < 4; ++st)
                xc[ut] = mmac(da[st], aok ? ld8(A_ + (16 * ut + pm) * RA + 16 * st + 4 * q)
                                          : v4bh{0, 0, 0, 0}, xc[ut]);
        }
    }
    #pragma unroll
    for (int ut = 0; ut < 4; ++ut)
        #pragma unroll
        for (int e = 0; e < 4; ++e) reinterpret_cast<short*>(XT)[(16 * ut + 4 * q + e) * kPT + tr] = bfr(xc[ut][e]);
    lds_barrier();                                                      // XT (and dwT in S) complete
    f4 dak[4];                       // F6: the lane's own dA row stays in registers (== s4b's dak)
    #pragma unroll
    for (int ut = 0; ut < 4; ++ut) {
        f4 acc = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int st = 0; st < 4; ++st) acc = mmac(aa[st], ld8(XT + (16 * ut + pm) * kPT + 16 * st + 4 * q), acc);
        #pragma unroll
        for (int e = 0; e < 4; ++e) { const int u = 16 * ut + 4 * q + e; dak[ut][e] = tr > u ? -acc[e] : 0.f; }
        *reinterpret_cast<f4*>(&lA[tr * 64 + 16 * ut + 4 * q]) = dak[ut];   // F6: for the transposes
    }
    // ================= phase B prologue =================
    s4bp_vmcnt<0>();                 // dAqk tile + every phase-A / prologue load
    lds_barrier();                   // dA and dAqk tiles visible
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
            akB[b - A][x] = keep ? lA[j * 64 + row] : 0.f;
            aqB[b - A][x] = (keep && g2_row_ok<M>(j, cl)) ? lQ[j * 64 + row] : 0.f;
        }
    v4bh avn[8];
    #pragma unroll
    for (int vt = 0; vt < 8; ++vt) avn[vt] = vn_native ? vraw[vt] : s4bp_c2a(vraw[vt], l);
    // R31: retire every loop-invariant global load visibly before the loop (no vmcnt re-waits in it)
    #pragma unroll
    for (int vt = 0; vt < 8; ++vt) asm volatile("" :: "v"(avn[vt]), "v"(ado[vt]));
    asm volatile("" :: "v"(db), "v"(beta_t));
    lds_barrier();                   // tiles consumed: stages reusable, phase-A scratch -> tables

    S4BPIO io;
    {
        const size_t cQ = g2_nat_blk(i, hh, H, 0, true, 8 * 8 * 256);
        io.rh = s4bp_srd(p.h + cQ, 8 * 8 * 256 * 2);
        io.rdh = s4bp_srd(p.dh + cQ, 8 * 8 * 256 * 2);
        io.rg = s4bp_srd(p.g + cb, (unsigned)(kBT * R * 4));
        io.rk = s4bp_srd(p.k + cb, (unsigned)(kBT * R * 2));
        io.rq = s4bp_srd(p.q_row + cb, (unsigned)(kBT * R * 2));
        io.rdq = s4bp_srd(p.dq + cb, (unsigned)(kBT * R * 4));
        io.rdk = s4bp_srd(p.dk + cb, (unsigned)(kBT * R * 4));
        io.rdg = s4bp_srd(p.dg + cb, (unsigned)(kBT * R * 4));
        io.oh = (unsigned)tid * 16u;
        const int gr = tid >> 2, gp = tid & 3;
        io.og = (unsigned)((g2_row_g<M>(gr, cl) * R + 4 * gp) * 4);
        const int u = tid & 127, kr = u >> 1, kh = u & 1;
        io.okq = g2_row_ok<M>(kr, cl) ? (unsigned)((kr * R + 8 * kh) * 2) : kS4bpOOB;
        io.rdw = io.rh; io.odw = 0u;                                    // unused (dwT is in S)
    }
    s4ab_issue<A>(io, sm.st[0], 0);
    s4ab_issue<A>(io, sm.st[1], 1);
    s4bp_vmcnt<NA>();                // stage 0 landed (stage 1 may be in flight)
    lds_barrier();

    // ================= phase B: the P-37 loop, dwT from S =================
    float dbacc = 0.f;
    #pragma unroll 1
    for (int kt = 0; kt < 8; ++kt) {
        const int buf = kt & 1;
        S4ABStage& Sg = sm.st[buf];
        const bf16* sh = &Sg.h[0];
        const bf16* sdh = &Sg.dh[0];
        auto HB = [&](int vt) { return ld8(sh + vt * 256 + l * 4); };
        auto DHB = [&](int vt) { return ld8(sdh + vt * 256 + l * 4); };
        const f4 gv = *reinterpret_cast<const f4*>(&Sg.g[row * 16 + 4 * q]);
        const f4 g8 = *reinterpret_cast<const f4*>(&Sg.g[(16 * A + 8) * 16 + 4 * q]);
        const v4bh kk = ld8(&Sg.k[row * 16 + 4 * q]);
        const v4bh qq = ld8(&Sg.q[row * 16 + 4 * q]);
        const f4 gn = *reinterpret_cast<const f4*>(&Sg.g[(kBT - 1) * 16 + 4 * q]);
        f4 dkp, dgK;
        {
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
                const float c = s4bp_msum(kd[e]);
                if (m == 0) sm.u.b.cs2[buf][A][4 * q + e] = c;
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
            if (q == 0) sm.u.b.cs1[buf][A][pm] = hd;
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
        f4 acc = {0.f, 0.f, 0.f, 0.f};       // dkgb = A^T (-dw): dwT rows kc..kc+15 straight from S
        #pragma unroll
        for (int st = 0; st < 4; ++st)
            acc = mmac(aa[st], ld8(S + (16 * kt + pm) * kPT + 4 * (4 * st + q)), acc);
        f4 ep, en;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float d = gv[e] - g8[e];
            ep[e] = ex2(d);
            en[e] = ex2(-d);
        }
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float kf = bf2f(kk[e]), qf = bf2f(qq[e]);
            const int o = (4 * e + q) * TS + row;
            sm.u.b.TA[o] = kf * en[e];
            sm.u.b.TQ[o] = qf * ep[e];
            sm.u.b.TK[o] = kf * beta_t * ep[e];
        }
        if (m == 8) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) sm.u.b.gM[A][4 * e + q] = gv[e];
        }
        lds_barrier();                                     // tables visible; stage kt consumed
        if (kt + 2 < 8) s4ab_issue<A>(io, Sg, kt + 2);

        f4 cq0 = {0.f, 0.f, 0.f, 0.f}, cq1 = cq0, ck0 = cq0, ck1 = cq0, ct0 = cq0, ct1 = cq0;
        {
        const float gm_n = sm.u.b.gM[A][m];
        #pragma unroll
        for (int b = 0; b <= A; ++b) {
            f4 tt = *reinterpret_cast<const f4*>(&sm.u.b.TA[m * TS + 16 * b + 4 * q]);
            if (b < A) tt *= ex2_diff(gm_n, sm.u.b.gM[b][m]);
            cq0 = mm8(lo2(aqA[b]), lo2(tt), cq0);
            ck0 = mm8(lo2(dak[b]), lo2(tt), ck0);
            cq1 = mm8(hi2(aqA[b]), hi2(tt), cq1);
            ck1 = mm8(hi2(dak[b]), hi2(tt), ck1);
        }
        #pragma unroll
        for (int b = A; b < 4; ++b) {
            f4 tq = *reinterpret_cast<const f4*>(&sm.u.b.TQ[m * TS + 16 * b + 4 * q]);
            f4 tk = *reinterpret_cast<const f4*>(&sm.u.b.TK[m * TS + 16 * b + 4 * q]);
            if (b > A) {
                const float s = ex2_diff(sm.u.b.gM[b][m], gm_n);
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
                const float s1 = ((sm.u.b.cs1[buf][0][c] + sm.u.b.cs1[buf][1][c]) + sm.u.b.cs1[buf][2][c]) + sm.u.b.cs1[buf][3][c];
                const float s2 = ((sm.u.b.cs2[buf][0][c] + sm.u.b.cs2[buf][1][c]) + sm.u.b.cs2[buf][2][c]) + sm.u.b.cs2[buf][3][c];
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
        s4bp_st3(io.rdq, io.rdk, io.rdg, so, oq, ok, og);
        // stage kt+1 complete before the barrier publishes it; ops issued after its group:
        // kt=0: stage 2 (4) + stores(0) (3); kt=1..5: stores(kt-1) + stage kt+2 + stores(kt); kt=6: 6
        if (kt == 0) s4bp_vmcnt<3 + NA>();
        else if (kt + 2 < 8) s4bp_vmcnt<6 + NA>();
        else if (kt + 1 < 8) s4bp_vmcnt<6>();
        lds_barrier();                                     // table WAR + stage kt+1 visible
    }
    db = xsum16(db) + xsum16(dbacc);
    if (q == 0 && rok) p.db[(size_t)(tok0 + row) * H + hh] = db;
}

template <bool VL, bool AFS>
__global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(1, 1)))
p3_wy_s4ab_kernel(WyS4AParams pa_, WyS4BParams p) {
    const WyS4AParams pa{g2_glb(pa_.kgb), g2_glb(pa_.v_vb), g2_glb(pa_.A), g2_glb(pa_.h), g2_glb(pa_.dv), g2_glb(pa_.beta), g2_glb(pa_.v_row), g2_glb(pa_.dv2), g2_glb(pa_.dwT), g2_glb(pa_.dA), g2_glb(pa_.db_part), pa_.T, pa_.H, pa_.NT, g2_glb(pa_.ctok), g2_glb(pa_.clen), g2_glb(pa_.dv2c)};
    __shared__ __align__(16) S4ABSmem sm;
    const int r = threadIdx.x >> 6;
    const int i = blockIdx.x;
    const long tok0 = g2_row_tok<VL>(p.ctok, i);
    const int cl = g2_row_len<VL>(p.clen, i);
    g2_vl_dispatch<VL>(cl, [&](auto mode_) {
        constexpr int M = decltype(mode_)::value;
        switch (__builtin_amdgcn_readfirstlane(r)) {
            case 0: s4ab_body<0, M, AFS>(pa, p, sm, tok0, cl); break;
            case 1: s4ab_body<1, M, AFS>(pa, p, sm, tok0, cl); break;
            case 2: s4ab_body<2, M, AFS>(pa, p, sm, tok0, cl); break;
            default: s4ab_body<3, M, AFS>(pa, p, sm, tok0, cl); break;
        }
    });
}
'''
anchor = 'extern "C" int g2_wy_s4_split_r('
assert t.count(anchor) == 1
t = t.replace(anchor, KERNEL + "\n" + anchor)

old = """    // T = chunk-padded rows (NT = T/64); with ctok/clen the row tensors are packed.
    if (T % kBT || NT != T / kBT || ((ctok == nullptr) != (clen == nullptr))) return -1;
    {
        WyS4AParams pa{"""
new = """    // T = chunk-padded rows (NT = T/64); with ctok/clen the row tensors are packed.
    if (T % kBT || NT != T / kBT || ((ctok == nullptr) != (clen == nullptr))) return -1;
    static const bool s4ab = [] { const char* e = getenv("KDA_G2_S4AB"); return e && atoi(e) == 1; }();
    if (s4ab) {   // kra P-39 (F6): s4a + s4b in one kernel; dwT / dA / db_part stay on chip
        WyS4AParams pa{(const bf16*)q_kgb, (const bf16*)v_vb, (const bf16*)A, (const bf16*)h,
                       (const bf16*)dv, (const float*)beta, (const bf16*)v_row,
                       (bf16*)dv2, (bf16*)dwT, (float*)dA, (float*)db_part, T, H, NT,
                       ctok, clen, (const bf16*)dv2c};
        WyS4BParams pb{(const bf16*)q_row, (const bf16*)k, (const bf16*)A, (const bf16*)h,
                       (const bf16*)do_, (const bf16*)dh, (const bf16*)v_new,
                       (const float*)g, (const float*)beta, (const float*)dAqk,
                       (const bf16*)dwT, (const float*)dA, (const float*)db_part,
                       (float*)dq, (float*)dk, (float*)dg, (float*)db, scale, T, H, NT,
                       ctok, clen, (const bf16*)do_row, (const bf16*)vnc};
        int afs_maxh = 12;
        if (const char* e = getenv("KDA_G2_S4A_AFS_MAXH")) afs_maxh = atoi(e);
        const bool afs = H <= afs_maxh;
        if (ctok) {
            if (afs) hipLaunchKernelGGL((p3_wy_s4ab_kernel<true, true>), dim3(NT, H), dim3(256), 0, s, pa, pb);
            else     hipLaunchKernelGGL((p3_wy_s4ab_kernel<true, false>), dim3(NT, H), dim3(256), 0, s, pa, pb);
        } else {
            if (afs) hipLaunchKernelGGL((p3_wy_s4ab_kernel<false, true>), dim3(NT, H), dim3(256), 0, s, pa, pb);
            else     hipLaunchKernelGGL((p3_wy_s4ab_kernel<false, false>), dim3(NT, H), dim3(256), 0, s, pa, pb);
        }
        return (int)hipGetLastError();
    }
    {
        WyS4AParams pa{"""
assert t.count(old) == 1, t.count(old)
t = t.replace(old, new)
P.write_text(t)
print("P-39 (F6: fused s4a+s4b) APPLIED")
