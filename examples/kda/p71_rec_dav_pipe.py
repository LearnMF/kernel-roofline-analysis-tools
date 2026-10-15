"""kra P-71 probe (plan A, consumer side, bitwise): dav runs on the CUs the TP8 recompute recurrence leaves idle,
chunk by chunk behind it.

C-5 (conc_bwd.py): with the recurrence on 48 CUs and dav on the other 32, ~214 us (16K/12) of dav work fits under the
recompute before hand-offs.  Structure (the recurrence is the PRODUCER, so it never waits):
  * the recompute instance publishes, every PUB steps and at its last step, prog[(bh*4 + vs)*4 + wv] = i + 1 from lane 0
    of each wave with a release store (agent scope) -- after that wave's Vn stores of chunks <= i;
  * dav part 1 (chunks [0, m), chunk-major grid) on a stream masked to the remaining CUs: each CTA's thread 0 spins
    with acquire loads until all 16 prog entries of its head are > i, then __syncthreads;
  * main stream joins both, dav part 2 (chunks [m, NT)) on all CUs writes the same outputs.
Same kernels, same per-element order -> bit-identical.  Python wiring: KDA_G2_BPIPE=1 (dense, no keep-state,
S4 split route), KDA_G2_BPIPE_M (part-1 chunks, default 0.9 NT), KDA_G2_BPIPE_PUB (16), KDA_G2_PIPE_RCU (48).
    python3 p71_rec_dav_pipe.py <tree>"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
g2 = root / "csrc" / "g2"


def sub(s, old, new, cnt=1):
    assert s.count(old) == cnt, (old[:80], s.count(old))
    return s.replace(old, new)


# ---------------------------------------------------------------- recurrence: progress publish
f = g2 / "g2_fwd_v8.cuh"
s = f.read_text()
s = sub(s, """    const float* GnRaw;                                // kra P-52: raw G_last (GE=false instance)
""", """    const float* GnRaw;                                // kra P-52: raw G_last (GE=false instance)
    int* prog = nullptr; int pub = 16;                  // kra P-71: per-wave progress publish
""")
head, rest = s.split("g2_fwd_h_v8o4_kernel(FwdOParams p) {", 1)
body, tail = rest.split("// kra P-49: g2_fwd_h_v8o4_kernel with NVH", 1)
anchor = """        if constexpr (PF) {   // kra P-64B: K prefetch after write_h
            __builtin_amdgcn_sched_barrier(0);
            loadK(nxt, kb0, kf0); loadK(nxt, kb1, kf1);
        }
"""
body = sub(body, anchor, anchor + """        if (p.prog != nullptr && (((i - c0 + 1) % p.pub) == 0 || i == c1 - 1)) {   // kra P-71
            if (l == 0)
                __hip_atomic_store(p.prog + ((size_t)bh * 4 + vs) * 4 + wv, i + 1, __ATOMIC_RELEASE,
                                   __HIP_MEMORY_SCOPE_AGENT);
        }
""")
s = head + "g2_fwd_h_v8o4_kernel(FwdOParams p) {" + body + "// kra P-49: g2_fwd_h_v8o4_kernel with NVH" + tail
s = sub(s, """extern "C" int g2_fwd_h_v8o(const void* Wn, const void* Kn, const void* Un, const void* Gn,""",
        """// kra P-71 probe plumbing
static int* g2_rp_prog = nullptr;
static int g2_rp_pub = 16;
extern "C" void g2_rec_prog_set(int* prog, int pub) { g2_rp_prog = prog; g2_rp_pub = pub > 0 ? pub : 16; }
extern "C" int g2_fwd_h_v8o(const void* Wn, const void* Kn, const void* Un, const void* Gn,""")
s = sub(s, """                 scale, H, T / kBT, seg, nseg, ctok, clen, (const float*)GnRaw};
""", """                 scale, H, T / kBT, seg, nseg, ctok, clen, (const float*)GnRaw};
    p.prog = g2_rp_prog; p.pub = g2_rp_pub;   // kra P-71
""")
f.write_text(s)

# ---------------------------------------------------------------- dav: chunk range / order / waits
d = g2 / "g2_dav.cuh"
s = d.read_text()
s = sub(s, """    const int* ctok; const int* clen;        // varlen phase 2 (null = dense)
};""", """    const int* ctok; const int* clen;        // varlen phase 2 (null = dense)
    const int* prog = nullptr; int c_off = 0, cm = 0, c_end = 0;   // kra P-71
};""")
s = sub(s, """    const int i = blockIdx.x, hh = blockIdx.y, tid = threadIdx.x;
    const int wv = tid >> 6, l = tid & 63;""", """    // kra P-71: cm 0 = head-major (production), 1 = chunk-major, 2 = 16-chunk blocks, head-major inside a block
    int i, hh;
    if (p_.cm == 2) {
        const int id = blockIdx.x, blk = id / (p_.H * 16), r = id - blk * (p_.H * 16);
        hh = r >> 4; i = p_.c_off + blk * 16 + (r & 15);
        if (i >= p_.c_end) return;
    } else {
        i = (p_.cm ? (int)blockIdx.y : (int)blockIdx.x) + p_.c_off; hh = p_.cm ? (int)blockIdx.x : (int)blockIdx.y;
    }
    const int tid = threadIdx.x;   // kra P-71
    if (p_.prog != nullptr) {      // kra P-71: wait until the recurrence published chunk i of head hh
        if (tid == 0) {
            #pragma unroll 1
            for (int e = 0; e < 16; ++e)
                while (__hip_atomic_load(p_.prog + (size_t)hh * 16 + e, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) <= i)
                    __builtin_amdgcn_s_sleep(2);
        }
        __syncthreads();
    }
    const int wv = tid >> 6, l = tid & 63;""")
s = sub(s, """extern "C" int g2_dav(const void* do_, const void* Vn, const void* A,""",
        """// kra P-71 probe plumbing
static int g2_dv_off = 0, g2_dv_n = -1, g2_dv_cm = 0;
static const int* g2_dv_prog = nullptr;
extern "C" void g2_dav_range_set(int off, int n, int cm, const int* prog) {
    g2_dv_off = off; g2_dv_n = n; g2_dv_cm = cm; g2_dv_prog = prog;
}
extern "C" int g2_dav(const void* do_, const void* Vn, const void* A,""")
s = sub(s, """                    (float)scale, H, T / 64, ctok, clen};
    if (ctok)
        hipLaunchKernelGGL(g2::g2_dav_kernel<true>, dim3(T / 64, H), dim3(256), 0, s, p);
    else
        hipLaunchKernelGGL(g2::g2_dav_kernel<false>, dim3(T / 64, H), dim3(256), 0, s, p);""",
        """                    (float)scale, H, T / 64, ctok, clen};
    p.prog = g2_dv_prog; p.c_off = g2_dv_off; p.cm = g2_dv_cm;   // kra P-71
    const int n = g2_dv_n >= 0 ? g2_dv_n : T / 64;
    if (n <= 0) return 0;
    p.c_end = g2_dv_off + n;
    const dim3 grid = g2_dv_cm == 2 ? dim3((unsigned)(H * ((n + 15) / 16) * 16), 1)
                                    : dim3(g2_dv_cm ? H : n, g2_dv_cm ? n : H);
    if (ctok)
        hipLaunchKernelGGL(g2::g2_dav_kernel<true>, grid, dim3(256), 0, s, p);
    else
        hipLaunchKernelGGL(g2::g2_dav_kernel<false>, grid, dim3(256), 0, s, p);""")
d.write_text(s)

# ---------------------------------------------------------------- bindings
b = g2 / "g2_bindings.cpp"
s = b.read_text()
s = sub(s, "extern \"C\" int g2_exp2_inplace(float* x, long n, hipStream_t s);   // kra P-52",
        "extern \"C\" int g2_exp2_inplace(float* x, long n, hipStream_t s);   // kra P-52\n"
        "extern \"C\" void g2_rec_prog_set(int* prog, int pub);   // kra P-71\n"
        "extern \"C\" void g2_dav_range_set(int off, int n, int cm, const int* prog);   // kra P-71")
anchor = "std::vector<torch::Tensor> dav(torch::Tensor do_, torch::Tensor Vn, torch::Tensor A,"
s = sub(s, anchor, """// kra P-71 probe: progress-publish switch for the next recurrence launch (empty tensor = off)
void rec_prog_set(torch::Tensor prog, int64_t pub) {
    g2_rec_prog_set(prog.numel() ? prog.data_ptr<int>() : nullptr, (int)pub);
}
// kra P-71 probe: dav over chunks [off, off+n) into caller-owned outputs (dense, no aN_do)
void dav_part(torch::Tensor do_, torch::Tensor Vn, torch::Tensor A, double scale, int64_t heads,
              int64_t chunks, torch::Tensor dA, torch::Tensor dVn, torch::Tensor dOT, int64_t off, int64_t n,
              int64_t cm, torch::Tensor prog) {
    check_contig(do_, "dav_part do"); check_contig(Vn, "dav_part Vn"); check_contig(A, "dav_part A");
    c10::hip::HIPGuard guard(do_.get_device());
    g2_dav_range_set((int)off, (int)n, (int)cm, prog.numel() ? prog.data_ptr<int>() : nullptr);
    const int rc = g2_dav(do_.data_ptr(), Vn.data_ptr(), A.data_ptr(), dA.data_ptr(), dVn.data_ptr(),
                          dOT.data_ptr(), nullptr, scale, 1, (int)(chunks * 64), (int)heads, nullptr, nullptr,
                          cur_stream());
    g2_dav_range_set(0, -1, 0, nullptr);
    TORCH_CHECK(rc == 0, "dav_part failed: ", rc);
}

""" + anchor)
s = sub(s, '    m.def("fwd_prep", &fwd_prep, "G2 R4: single-pass forward prep (KDA_G2_FI=4)");',
        '    m.def("fwd_prep", &fwd_prep, "G2 R4: single-pass forward prep (KDA_G2_FI=4)");\n'
        '    m.def("rec_prog_set", &rec_prog_set, "kra P-71 probe");\n'
        '    m.def("dav_part", &dav_part, "kra P-71 probe");')
b.write_text(s)

# ---------------------------------------------------------------- python wiring
k = root / "hip_kda" / "ops" / "k3_g2.py"
s = k.read_text()
s = sub(s, """def nofla() -> bool:""", """_PIPE_STREAMS = {}


def _pipe_streams(rcu):
    \"\"\"kra P-71 probe: (recurrence stream on CUs [0, rcu), partner stream on the rest), CU-masked.\"\"\"
    if rcu not in _PIPE_STREAMS:
        import ctypes
        hip = ctypes.CDLL("libamdhip64.so")
        ncu = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        nw = (ncu + 31) // 32
        out = []
        for lo, hi in ((0, rcu), (rcu, ncu)):
            m = (ctypes.c_uint32 * nw)()
            for c in range(lo, hi):
                m[c // 32] |= 1 << (c % 32)
            st = ctypes.c_void_p()
            assert hip.hipExtStreamCreateWithCUMask(ctypes.byref(st), ctypes.c_uint32(nw), m) == 0
            out.append(torch.cuda.ExternalStream(st.value))
        _PIPE_STREAMS[rcu] = tuple(out)
    return _PIPE_STREAMS[rcu]


def nofla() -> bool:""")
old = """        if kept:                                   # R5 keep-state: bit-identical to the recompute
            bN_h, Vn, aN_vn = kept_bN, kept_Vn, kept_aN
            del kept_bN, kept_Vn, kept_aN
        else:"""
new = """        no_ando = no_vb and os.environ.get("KDA_G2_NOADO", "1").strip() != "0"
        bpipe = (os.environ.get("KDA_G2_BPIPE", "0") == "1" and not kept and lay is None and ctx.seg is None
                 and no_ando and no_an)
        if bpipe:   # kra P-71 probe: recompute (48 CUs) || dav part 1 (the rest), dav part 2 after the join
            ext = g2_ops._ext()
            m = min(NT, max(0, int(os.environ.get("KDA_G2_BPIPE_M", str(round(0.9 * NT))))))
            pub = int(os.environ.get("KDA_G2_BPIPE_PUB", "16"))
            sA, sB = _pipe_streams(int(os.environ.get("KDA_G2_PIPE_RCU", "48")))
            main = torch.cuda.current_stream()
            prog = torch.zeros(H * 16, dtype=torch.int32, device=do.device)
            none_i = torch.empty(0, dtype=torch.int32, device=do.device)
            dAqk = torch.empty(1, 64 * NT, H, 64, dtype=torch.float32, device=do.device)
            dVn = torch.empty(1, H, NT, 4, 8, 4, 16, 4, dtype=do.dtype, device=do.device)
            dOT = torch.empty(1, H, NT, 8, 4, 4, 16, 4, dtype=do.dtype, device=do.device)
            do_c = do.contiguous()
            ev = main.record_event()
            sA.wait_event(ev); sB.wait_event(ev)
            with torch.cuda.stream(sA):
                ext.rec_prog_set(prog, pub)
                bN_h, Vn, aN_vn = g2_ops.fwd_h_bn(Wn, Kn, Un, Gn_rec, seg=ctx.seg, emit_an=False, gn_raw=Gn_raw)
                ext.rec_prog_set(none_i, 0)
            with torch.cuda.stream(sB):
                ext.dav_part(do_c, Vn, Aqk, scale, H, NT, dAqk, dVn, dOT, 0, m,
                             int(os.environ.get("KDA_G2_BPIPE_CM", "2")), prog)
            main.wait_stream(sA); main.wait_stream(sB)
            ext.dav_part(do_c, Vn, Aqk, scale, H, NT, dAqk, dVn, dOT, m, NT - m, 0, none_i)
            aN_do = torch.empty(0, dtype=do.dtype, device=do.device)
            vn_c, aN_vn = Vn, torch.empty(0, dtype=Vn.dtype, device=Vn.device)
            del Wn, Kn, Un, Vn
        elif kept:                                   # R5 keep-state: bit-identical to the recompute
            bN_h, Vn, aN_vn = kept_bN, kept_Vn, kept_aN
            del kept_bN, kept_Vn, kept_aN
        else:"""
s = sub(s, old, new)
old2 = """        del Wn, Kn, Un
        vn_c = None
        if not kept and no_an:
            vn_c, aN_vn = Vn, torch.empty(0, dtype=Vn.dtype, device=Vn.device)
"""
new2 = """        if not bpipe:
            del Wn, Kn, Un
            vn_c = None
            if not kept and no_an:
                vn_c, aN_vn = Vn, torch.empty(0, dtype=Vn.dtype, device=Vn.device)
"""
s = sub(s, old2, new2)
old3 = """        no_ando = no_vb and os.environ.get("KDA_G2_NOADO", "1").strip() != "0"
        dAqk, dVn, dOT, aN_do = g2_ops.dav(do, Vn, Aqk, scale, H, NT, rowmap=lay,
                                           emit_ando=not no_ando)
        if no_ando:
            aN_do = torch.empty(0, dtype=do.dtype, device=do.device)
        del Vn                                                    # reads native Vn,"""
new3 = """        if not bpipe:
            dAqk, dVn, dOT, aN_do = g2_ops.dav(do, Vn, Aqk, scale, H, NT, rowmap=lay,
                                               emit_ando=not no_ando)
            if no_ando:
                aN_do = torch.empty(0, dtype=do.dtype, device=do.device)
            del Vn                                                # reads native Vn,"""
s = sub(s, old3, new3)
k.write_text(s)
print("P-71 APPLIED")
