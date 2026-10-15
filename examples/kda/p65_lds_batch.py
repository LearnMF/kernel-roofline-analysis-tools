"""kra P-65 probe (bitwise, scheduling only): KP (H <= 20) recurrence instances issue each GEMM's LDS operand reads
as one batch ahead of its MMAC chain.

SQTT of the recompute step (cp_p62_d3): ~500 of ~2300 cycles are ds_read2_b64 latency -- GEMM1 after barrier A and
GEMM2 after barrier B each wait 4 times ~60-95 cycles because the compiler interleaves "read, wait, 2 MMACs".  With
every read of a GEMM issued first (sched_barrier fence before the MMAC loop) one LDS latency is exposed per GEMM and
the reads stream at LDS bandwidth.  Same operands, same MMAC order -> bit-identical.
    python3 p65_lds_batch.py <tree>"""
import sys
from pathlib import Path

f = Path(sys.argv[1]) / "csrc" / "g2" / "g2_fwd_v8.cuh"
s = f.read_text()
head, rest = s.split("g2_fwd_h_v8o4_kernel(FwdOParams p) {", 1)
body, tail = rest.split("// kra P-49: g2_fwd_h_v8o4_kernel with NVH", 1)


def sub(b, old, new):
    assert b.count(old) == 1, (old[:80], b.count(old))
    return b.replace(old, new)


body = sub(body, """        #pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
            hbf0[kt] = ld8(hB + (0 * 16 + (VP ? pi16(m) : m)) * kPadK + 16 * kt + q4 * 4);   // kra P-62
            hbf1[kt] = ld8(hB + (1 * 16 + (VP ? pi16(m) : m)) * kPadK + 16 * kt + q4 * 4);
            acc0 = mmac(wf[kt], hbf0[kt], acc0);
            acc1 = mmac(wf[kt], hbf1[kt], acc1);
        }""", """        if constexpr (KP) {   // kra P-65: all 16 hB fragments first, then the MMAC chain
            #pragma unroll
            for (int kt = 0; kt < 8; ++kt) {
                hbf0[kt] = ld8(hB + (0 * 16 + (VP ? pi16(m) : m)) * kPadK + 16 * kt + q4 * 4);
                hbf1[kt] = ld8(hB + (1 * 16 + (VP ? pi16(m) : m)) * kPadK + 16 * kt + q4 * 4);
            }
            __builtin_amdgcn_sched_barrier(0);
            #pragma unroll
            for (int kt = 0; kt < 8; ++kt) {
                acc0 = mmac(wf[kt], hbf0[kt], acc0);
                acc1 = mmac(wf[kt], hbf1[kt], acc1);
            }
        } else {
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
            hbf0[kt] = ld8(hB + (0 * 16 + (VP ? pi16(m) : m)) * kPadK + 16 * kt + q4 * 4);   // kra P-62
            hbf1[kt] = ld8(hB + (1 * 16 + (VP ? pi16(m) : m)) * kPadK + 16 * kt + q4 * 4);
            acc0 = mmac(wf[kt], hbf0[kt], acc0);
            acc1 = mmac(wf[kt], hbf1[kt], acc1);
        }
        }""")
old_g2 = """        // ---- GEMM2 for both K-blocks
        #pragma unroll
        for (int vt = 0; vt < 2; ++vt) {
            f4 upd = f4{0.f,0.f,0.f,0.f};
            #pragma unroll
            for (int rt = 0; rt < 4; ++rt) {
                const v4bh va = ld8(vB + (16 * vt + hrow) * kPadT + 16 * rt + q4 * 4);   // kra P-63
                upd = mmac(va, kf0[rt], upd);
            }
            ht0[vt] += upd;
        }
        #pragma unroll
        for (int vt = 0; vt < 2; ++vt) {
            f4 upd = f4{0.f,0.f,0.f,0.f};
            #pragma unroll
            for (int rt = 0; rt < 4; ++rt) {
                const v4bh va = ld8(vB + (16 * vt + hrow) * kPadT + 16 * rt + q4 * 4);   // kra P-63
                upd = mmac(va, kf1[rt], upd);
            }
            ht1[vt] += upd;
        }"""
new_g2 = """        // ---- GEMM2 for both K-blocks
        if constexpr (KP) {   // kra P-65: the 8 vB fragments (shared by both K-blocks) first, then the MMACs
            v4bh vaf[2][4];
            #pragma unroll
            for (int vt = 0; vt < 2; ++vt)
                #pragma unroll
                for (int rt = 0; rt < 4; ++rt)
                    vaf[vt][rt] = ld8(vB + (16 * vt + hrow) * kPadT + 16 * rt + q4 * 4);
            __builtin_amdgcn_sched_barrier(0);
            #pragma unroll
            for (int vt = 0; vt < 2; ++vt) {
                f4 upd = f4{0.f,0.f,0.f,0.f};
                #pragma unroll
                for (int rt = 0; rt < 4; ++rt) upd = mmac(vaf[vt][rt], kf0[rt], upd);
                ht0[vt] += upd;
            }
            #pragma unroll
            for (int vt = 0; vt < 2; ++vt) {
                f4 upd = f4{0.f,0.f,0.f,0.f};
                #pragma unroll
                for (int rt = 0; rt < 4; ++rt) upd = mmac(vaf[vt][rt], kf1[rt], upd);
                ht1[vt] += upd;
            }
        } else {
""" + old_g2.split("// ---- GEMM2 for both K-blocks\n", 1)[1] + "\n        }"
body = sub(body, old_g2, new_g2)
f.write_text(head + "g2_fwd_h_v8o4_kernel(FwdOParams p) {" + body + "// kra P-49: g2_fwd_h_v8o4_kernel with NVH" + tail)
print("P-65 APPLIED")
