"""kra P-70b probe (on top of P-70): CU-partitioned overlap.  P-70 measured that when the tail prep and the recurrence
become runnable together, prep CTAs land on the recurrence's CUs and the chain-bound recurrence runs ~2x slower
(trace: 1021 vs 518 us), so the overlap loses ~300 us.  With KDA_G2_PIPE_MASK=1 the recurrence runs on a stream whose
CU mask holds KDA_G2_PIPE_RCU (48) CUs and the tail prep on a second stream masked to the remaining CUs
(hipExtStreamCreateWithCUMask); the main stream waits for both.  Bit-identical (scheduling only).
    python3 p70b_cumask.py <tree>"""
import sys
from pathlib import Path

b = Path(sys.argv[1]) / "csrc" / "g2" / "g2_bindings.cpp"
s = b.read_text()


def sub(s, old, new):
    assert s.count(old) == 1, (old[:80], s.count(old))
    return s.replace(old, new)


s = sub(s, """    static hipStream_t side = nullptr;
    if (side == nullptr) TORCH_CHECK(hipStreamCreateWithFlags(&side, hipStreamNonBlocking) == hipSuccess, "side stream");""",
        """    static hipStream_t side_plain = nullptr, side_m = nullptr, tail_m = nullptr;
    const char* em = getenv("KDA_G2_PIPE_MASK");
    const bool masked = em != nullptr && atoi(em) != 0;   // kra P-70b
    if (!masked && side_plain == nullptr)
        TORCH_CHECK(hipStreamCreateWithFlags(&side_plain, hipStreamNonBlocking) == hipSuccess, "side stream");
    if (masked && side_m == nullptr) {
        int dev = 0, ncu = 0;
        hipGetDevice(&dev);
        hipDeviceGetAttribute(&ncu, hipDeviceAttributeMultiprocessorCount, dev);
        const char* er = getenv("KDA_G2_PIPE_RCU");
        const int rcu = er ? atoi(er) : 48;
        uint32_t ma[8] = {0}, mb[8] = {0};
        for (int c = 0; c < ncu && c < 256; ++c) (c < rcu ? ma : mb)[c / 32] |= 1u << (c % 32);
        const uint32_t nw = (uint32_t)((ncu + 31) / 32);
        TORCH_CHECK(hipExtStreamCreateWithCUMask(&side_m, nw, ma) == hipSuccess, "masked side stream");
        TORCH_CHECK(hipExtStreamCreateWithCUMask(&tail_m, nw, mb) == hipSuccess, "masked tail stream");
    }
    hipStream_t side = masked ? side_m : side_plain;""")
s = sub(s, """    auto prep = [&](int off, int n, int* fl) {""", """    auto prep = [&](int off, int n, int* fl, hipStream_t st) {""")
s = sub(s, """                                   1, scale, (int)H, (int)NT, nullptr, nullptr, main);""",
        """                                   1, scale, (int)H, (int)NT, nullptr, nullptr, st);""")
s = sub(s, """    prep(0, i_tail, nullptr);""", """    prep(0, i_tail, nullptr, main);""")
s = sub(s, """    prep(i_tail, (int)k_tail, flags.data_ptr<int>());   // the tail on the main stream, concurrently
    hipEventRecord(e1, side);
    hipStreamWaitEvent(main, e1, 0);
    hipEventDestroy(e0); hipEventDestroy(e1);""", """    hipEvent_t e2;
    hipEventCreateWithFlags(&e2, hipEventDisableTiming);
    if (masked) {   // kra P-70b: the tail on the complementary CUs
        hipStreamWaitEvent(tail_m, e0, 0);
        prep(i_tail, (int)k_tail, flags.data_ptr<int>(), tail_m);
        hipEventRecord(e2, tail_m);
        hipStreamWaitEvent(main, e2, 0);
    } else {
        prep(i_tail, (int)k_tail, flags.data_ptr<int>(), main);   // the tail on the main stream, concurrently
    }
    hipEventRecord(e1, side);
    hipStreamWaitEvent(main, e1, 0);
    hipEventDestroy(e0); hipEventDestroy(e1); hipEventDestroy(e2);""")
b.write_text(s)
print("P-70b APPLIED")
