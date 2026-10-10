"""Project C feasibility probe (NOT for production, values not checked): s4b keeps its row's
dq (and optionally dk) in registers across the kt loop (fully unrolled), computes the l2norm-
backward row dot with the bwd_tail summation tree (pairs -> lane bits q -> kt bits) and writes
bf16 dx directly.  Purpose: measure VGPR / scratch / time of the "row-complete" s4b that the
full s4b+tail fusion needs.  Usage: python3 c_probe_s4b_rowc.py <g2 dir> [--dk]"""
import sys
from pathlib import Path

DK = "--dk" in sys.argv
p = Path(sys.argv[1]) / "g2_wy.cuh"
t = p.read_text()
a = t.index("template <int A, int M, bool L2R>\n__device__ __forceinline__ void s4b_tail(")
b = t.index("\n}\n", a)
body = t[a:b]
old_loop = "    float dbacc = 0.f;\n    #pragma unroll 1\n    for (int kt = 0; kt < 8; ++kt) {"
assert body.count(old_loop) == 1
body = body.replace(old_loop, "    float dbacc = 0.f;\n    f4 dqs[8];" + (" f4 dks[8];" if DK else "") +
                    "   // C probe: row-complete outputs held across kt\n    #pragma unroll\n    for (int kt = 0; kt < 8; ++kt) {")
old_st = """            *reinterpret_cast<f4*>(dq_ + ro + kc) = oq;
            *reinterpret_cast<f4*>(dk_ + ro + kc) = ok;"""
assert body.count(old_st) == 1
body = body.replace(old_st, ("" if DK else "            *reinterpret_cast<f4*>(dk_ + ro + kc) = ok;\n").rstrip("\n") if not DK else "")
body = body.replace("        dbacc += (dbp[0] + dbp[1]) + (dbp[2] + dbp[3]);",
                    "        dbacc += (dbp[0] + dbp[1]) + (dbp[2] + dbp[3]);\n        dqs[kt] = oq;" + (" dks[kt] = ok;" if DK else ""))
tail = """
    // C probe: l2norm backward on the held rows -- tree: pairs (e) -> lane bits (q) -> kt bits
    auto rowdx = [&](f4 (&dy)[8], const bf16* y_, float r, float* outbase) {
        float s[8];
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
            const v4bh yy = rok ? ld8(y_ + ro + 16 * kt) : v4bh{0, 0, 0, 0};
            const float p0 = dy[kt][0] * bf2f(yy[0]) + dy[kt][1] * bf2f(yy[1]);
            const float p1 = dy[kt][2] * bf2f(yy[2]) + dy[kt][3] * bf2f(yy[3]);
            float v = p0 + p1;
            v += __shfl_xor(v, 16, 64);
            v += __shfl_xor(v, 32, 64);
            s[kt] = v;
        }
        const float dot = ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
            const v4bh yy = rok ? ld8(y_ + ro + 16 * kt) : v4bh{0, 0, 0, 0};
            v4bh o;
            #pragma unroll
            for (int e = 0; e < 4; ++e) o[e] = bf_rne(dy[kt][e] * r - dot * bf2f(yy[e]) * r);
            if (rok) *reinterpret_cast<v4bh*>(reinterpret_cast<short*>(outbase) + ro + 16 * kt) = o;
        }
    };
    rowdx(dqs, q_, rq, dq_);
""" + ("    rowdx(dks, k_, rk, dk_);\n" if DK else "")
body = body + tail
t = t[:a] + body + t[b:]
p.write_text(t)
print("C PROBE APPLIED", "(dq+dk)" if DK else "(dq)")
