"""kra P-77b (bitwise): the forward prep emits qn / kn once and saves them instead of the raw q / k, so the backward
needs no l2n_apply and none of its consumers change (P-77 showed folding the l2n VALU into wu / s4b costs more than
the launch it removes, R57).

prep_a already holds bf_rne(bf2f(x) * rstd) for its 4 rows x 8 columns per lane (l2n_apply's exact expression, same
rstd) -- two 8 B stores per row per tensor write them row-major [T,H,128].  Saved-tensor bytes are unchanged (qn/kn
replace q/k); total memory is unchanged when the caller does not otherwise retain q/k (Megatron-Bridge K3: q/k are
FLA ShortConvolution outputs, whose autograd saves the conv INPUT -- fla/modules/conv/*/ops.py).
KDA_G2_SAVE_QKN=0 restores the raw save + backward l2n_apply.  Only the R4 prep route (FI=4) with in-kernel l2norm.
    python3 p77b_save_qkn.py <tree>"""
import sys
from pathlib import Path

R = Path(sys.argv[1])
G = R / "csrc" / "g2"


def sub(path, old, new, cnt=1):
    s = path.read_text()
    assert s.count(old) == cnt, (path.name, old[:90], s.count(old))
    path.write_text(s.replace(old, new))


fp = G / "g2_fwd_prep.cuh"
sub(fp, """    int kp = 0;                         // kra P-61e: Kn lane-permuted + Gn natural-k (H <= KPERM_MAX_H)
};""", """    int kp = 0;                         // kra P-61e: Kn lane-permuted + Gn natural-k (H <= KPERM_MAX_H)
    short *qn_out = nullptr, *kn_out = nullptr;   // kra P-77b: non-null -> emit qn / kn row-major
};""")
sub(fp, """            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                qv[rb][u >> 2][u & 3] = bf_rne(bf2f(qv[rb][u >> 2][u & 3]) * rq);
                kv[rb][u >> 2][u & 3] = bf_rne(bf2f(kv[rb][u >> 2][u & 3]) * rk);
            }
""", """            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                qv[rb][u >> 2][u & 3] = bf_rne(bf2f(qv[rb][u >> 2][u & 3]) * rq);
                kv[rb][u >> 2][u & 3] = bf_rne(bf2f(kv[rb][u >> 2][u & 3]) * rk);
            }
            if (p.qn_out != nullptr && g2_row_ok<M>(row, cl)) {   // kra P-77b: qn / kn for the backward
                const size_t o2 = cta0 + (size_t)row * rs + cbase;
                *reinterpret_cast<v4bh*>(p.qn_out + o2) = qv[rb][0];
                *reinterpret_cast<v4bh*>(p.qn_out + o2 + 4) = qv[rb][1];
                *reinterpret_cast<v4bh*>(p.kn_out + o2) = kv[rb][0];
                *reinterpret_cast<v4bh*>(p.kn_out + o2 + 4) = kv[rb][1];
            }
""")
sub(fp, """                           double lower_bound, double gscale, int gate, double scale, int H,
                           int NT, const int* ctok, const int* clen, hipStream_t s) {
    if ((ctok == nullptr) != (clen == nullptr) || NT <= 0) return -1;""",
    """                           double lower_bound, double gscale, int gate, double scale, int H,
                           int NT, const int* ctok, const int* clen, void* qn_out, void* kn_out,
                           hipStream_t s) {   // kra P-77b: qn_out / kn_out
    if ((ctok == nullptr) != (clen == nullptr) || NT <= 0) return -1;""")
sub(fp, """    pa.kp = H <= g2::g2_env_long("KDA_G2_KPERM_MAX_H", 20);   // kra P-61e
    const dim3 grid(NT, H), blk(256);""", """    pa.kp = H <= g2::g2_env_long("KDA_G2_KPERM_MAX_H", 20);   // kra P-61e
    pa.qn_out = (short*)qn_out; pa.kn_out = (short*)kn_out;    // kra P-77b
    const dim3 grid(NT, H), blk(256);""")

bd = G / "g2_bindings.cpp"
sub(bd, """                           double lower_bound, double gscale, int gate, double scale, int H,
                           int NT, const int* ctok, const int* clen, hipStream_t s);""",
    """                           double lower_bound, double gscale, int gate, double scale, int H,
                           int NT, const int* ctok, const int* clen, void* qn_out, void* kn_out,
                           hipStream_t s);   // kra P-77b""")
sub(bd, """                                    c10::optional<torch::Tensor> ctok,
                                    c10::optional<torch::Tensor> clen) {
    TORCH_CHECK(q.dim() == 4 && q.size(0) == 1 && q.size(3) == 128, "g2_fwd_prep: q [1,T,H,128]");""",
    """                                    c10::optional<torch::Tensor> ctok,
                                    c10::optional<torch::Tensor> clen, bool emit_qkn) {   // kra P-77b
    TORCH_CHECK(q.dim() == 4 && q.size(0) == 1 && q.size(3) == 128, "g2_fwd_prep: q [1,T,H,128]");""")
sub(bd, """    auto Kb = torch::empty_like(q);
    const int rc = g2_fwd_prep(q.data_ptr(), k.data_ptr(), v.data_ptr(), g_raw.data_ptr(),""",
    """    auto Kb = torch::empty_like(q);
    auto qn = emit_qkn ? torch::empty_like(q) : torch::Tensor();   // kra P-77b
    auto kn = emit_qkn ? torch::empty_like(k) : torch::Tensor();
    const int rc = g2_fwd_prep(q.data_ptr(), k.data_ptr(), v.data_ptr(), g_raw.data_ptr(),""")
sub(bd, """                               rm.ctok, rm.clen, cur_stream());
    TORCH_CHECK(rc == 0, "g2_fwd_prep failed: ", rc);
    return {Wn, Kn, Un, QgA, AqkA, Aqk, Akk, qr, kr, Gn};""",
    """                               rm.ctok, rm.clen, emit_qkn ? qn.data_ptr() : nullptr,
                               emit_qkn ? kn.data_ptr() : nullptr, cur_stream());
    TORCH_CHECK(rc == 0, "g2_fwd_prep failed: ", rc);
    return {Wn, Kn, Un, QgA, AqkA, Aqk, Akk, qr, kr, Gn, qn, kn};""")

po = R / "hip_kda" / "ops" / "g2_ops.py"
sub(po, """def fwd_prep(q, k, v, g_raw, a_log, dt_bias, beta, lower_bound, gscale, gate, scale, heads,
             chunks, rowmap=None):""", """def fwd_prep(q, k, v, g_raw, a_log, dt_bias, beta, lower_bound, gscale, gate, scale, heads,
             chunks, rowmap=None, emit_qkn=False):""")
sub(po, """    return _ext().fwd_prep(q.contiguous(), k.contiguous(), v.contiguous(), g_raw.contiguous(),
                           a_log, dt_bias, beta.contiguous(), float(lower_bound or 0.0),
                           float(gscale), bool(gate), float(scale), int(heads), int(chunks), ct, cl)""",
    """    r = _ext().fwd_prep(q.contiguous(), k.contiguous(), v.contiguous(), g_raw.contiguous(),
                        a_log, dt_bias, beta.contiguous(), float(lower_bound or 0.0),
                        float(gscale), bool(gate), float(scale), int(heads), int(chunks), ct, cl,
                        bool(emit_qkn))
    return list(r) if emit_qkn else list(r)[:10]   # kra P-77b: + (qn, kn) when emitted""")

k3 = R / "hip_kda" / "ops" / "k3_g2.py"
sub(k3, """            gate = not precomputed_g
            Wn, Kn, Un, QgA, AqkA, Aqk, Akk, q_rstd, k_rstd, Gn = g2_ops.fwd_prep(
                q, k, v, g_raw, a_log.float() if gate else None, dt_bias.float() if gate else None,
                beta, lower_bound if gate else 0.0, RCP_LN2, gate, scale, H, NT, rowmap=lay)""",
    """            gate = not precomputed_g
            # kra P-77b: save qn/kn (emitted by prep_a) instead of q/k -> no backward l2n_apply
            sqkn = (any(ctx.needs_input_grad)
                    and os.environ.get("KDA_G2_SAVE_QKN", "1").strip() != "0"
                    and os.environ.get("KDA_G2_BL2", "1").strip() != "0")
            pr = g2_ops.fwd_prep(
                q, k, v, g_raw, a_log.float() if gate else None, dt_bias.float() if gate else None,
                beta, lower_bound if gate else 0.0, RCP_LN2, gate, scale, H, NT, rowmap=lay,
                emit_qkn=sqkn)
            Wn, Kn, Un, QgA, AqkA, Aqk, Akk, q_rstd, k_rstd, Gn = pr[:10]
            q_sv, k_sv = (pr[10], pr[11]) if sqkn else (q, k)
            ctx.qkn = sqkn
            del pr""")
sub(k3, """            ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g_raw, beta_raw, beta,
                                  a_log, dt_bias, Aqk, Akk, *kept)""",
    """            ctx.save_for_backward(q_sv, q_rstd, k_sv, k_rstd, v, g_raw, beta_raw, beta,
                                  a_log, dt_bias, Aqk, Akk, *kept)   # kra P-77b: qn/kn when ctx.qkn""")
sub(k3, """        qn, kn = q, k
        if ctx.use_qk_l2norm_in_kernel:
            if os.environ.get("KDA_G2_BL2", "1").strip() != "0":""",
    """        qn, kn = q, k                     # kra P-77b: already qn/kn when ctx.qkn
        if ctx.use_qk_l2norm_in_kernel and not getattr(ctx, "qkn", False):
            if os.environ.get("KDA_G2_BL2", "1").strip() != "0":""")
print("P-77b APPLIED")
