"""First-principles lower-bound model of KDA fwd+bwd (chunked gated delta rule with per-channel decay),
independent of any implementation's kernel split (kra methodology 1.3.4: compute the hardware limit
first, then attribute the measured gap to model omissions vs implementation overhead).

Algorithm (FLA chunk_kda semantics, use_qk_l2norm_in_kernel, gate g = lb*sigmoid(A*(g_raw+dt)) in
the log2 domain, safe gate, chunk C, d = d_k = d_v):
  per chunk c (rows i,j in [0,C), per-channel chunk-local cumulative gate G):
    Akk[i,j] = beta_i * sum_d k_i k_j exp2(G_i - G_j)  (j < i);  T = (I + Akk)^-1
    W = T (beta k exp2(G)),  U = T (beta v)
    Aqk[i,j] = scale * sum_d q_i k_j exp2(G_i - G_j)    (j <= i)
    v_new = U - W h_c;   o = scale (q exp2(G)) h_c + Aqk v_new
    h_{c+1} = exp2(G_last) * h_c + (k exp2(G_last - G))^T v_new
  backward: the same chunks in reverse (dh recurrence) + per-chunk gradients, gate / l2norm / beta
  backward.  Memory policy P1 (as hip_kda and FLA default): the forward keeps rstd, Aqk, T (Akk);
  the backward recomputes the chunk states (one extra sequential pass).  P2: the forward also keeps
  bf16 chunk states h (+512 B/token*head, == FLA-default budget 1032 B), no recompute pass.

Four bounds per (machine, T, H):  T_mem (irreducible HBM bytes), T_mmac (algorithmic matrix FLOPs),
T_valu (elementwise lane-ops + transcendentals on the vector pipe), T_chain (sequential chunk steps x
the minimal per-step latency).  Perfect overlap -> max(...); the chain bracket adds T_chain to the
largest throughput term.

    python3 kda_bound.py [--policy P1|P2] [--lstep-us X] [--measured]   (prints markdown)"""
import argparse
import json
import math

C, D = 64, 128


# ---------------------------------------------------------------- work inventory (per token*head)
def inventory(policy="P1"):
    """Algorithmic work per token*head.  MACs counted with triangular savings where the operator is
    triangular; elementwise counted per channel; every entry is (fwd, bwd)."""
    tri = C * C * D / 2                     # one lower-triangular C x C x D product
    sq = C * D * D                          # one C x D x D product
    t3 = C ** 3 / 6                         # triangular inverse of a C x C matrix
    mac = {
        # forward: Akk, Aqk, T, W, U (tri) + v_new, o1, h (square) + o2 (tri)
        "fwd": 4 * tri + t3 + tri + 3 * sq,
        # backward: dAqk, dv_new(o), dq~(Aqk), dk~(Aqk), dT x2, dK~, dV, dAkk->dk x2 (tri, 10) ; dAkk (t3 x2)
        #           dv2, dh(q~), dh(W), dq~(h), dk_bar(h), dW (square, 6)
        "bwd": 10 * tri + 2 * t3 + 6 * sq,
        # P1 recompute in the backward: W, U (tri x2) + v_new, h (square x2)
        "recompute": 2 * tri + 2 * sq,
    }
    mac = {k: v / C for k, v in mac.items()}   # the products above are per chunk (C tokens)
    # elementwise vector work per token (lane-ops) and transcendentals (exp2 / rcp / rsqrt)
    valu = {
        "fwd": (4 * D            # l2norm q,k: sumsq + scale
                + 5 * D + D      # gate: add, mul, add, mul(lb), mul(log2e) + bf16->f32
                + 4 * D          # chunk-local cumsum (Kahan-compensated, as the reference kernels)
                + 6 * D          # scaled bf16 operands: q~, k~ (beta), k_bar(G_last), + cvt
                + 2 * D + 2 * D  # v_new subtract + cvt;  o: fma + cvt
                + D * D / C),    # state decay h *= exp2(G_last), once per chunk
        "bwd": (2 * D            # l2norm apply (rstd kept)
                + 6 * D + 4 * D  # gate + cumsum recomputed (policy P1 keeps no gate)
                + 6 * D          # scaled operands again
                + 30 * D         # dq / dk / dg / dbeta chain rule through the decays, conversions
                + D * D / C      # dh decay per chunk
                + 4 * D          # reverse cumsum of dg (Kahan)
                + 6 * D          # gate backward (sigmoid derivative chain, bf16 round, ddt partial)
                + 8 * D),        # l2norm backward q,k (dot + 2 scales each)
        "recompute": 2 * D + D * D / C,   # v_new + state decay in the recompute pass
    }
    sfu = {
        "fwd": 2 + 2 * D + 3 * D,   # rsqrt x2, gate exp+rcp, decay exp2 (G, -G, G_last-G)
        "bwd": 2 * D + 3 * D + 2 * D,   # gate recompute, decays recompute, gate-backward exp+rcp
        "recompute": 0,
    }
    # irreducible HBM bytes (bf16 = 2, fp32 = 4)
    rb = {
        "fwd": 4 * 2 * D + 4,                       # q k v g_raw + beta
        "bwd": 4 * 2 * D + 4 + 2 * D + (8 + 2 * C + 2 * C),   # inputs + do + saved (rstd, Aqk, T)
    }
    wb = {
        "fwd": 2 * D + (8 + 2 * C + 2 * C),          # o + saved
        "bwd": 4 * 2 * D + 4,                        # dq dk dv dg_raw + dbeta
    }
    passes = 3
    if policy == "P2":                               # keep bf16 chunk states: + write fwd, + read bwd
        wb["fwd"] += 2 * D * D / C
        rb["bwd"] += 2 * D * D / C
        mac["recompute"] = 0
        valu["recompute"] = 0
        passes = 2
    return mac, valu, sfu, rb, wb, passes


# ---------------------------------------------------------------- machines (measured where possible)
MACHINES = {
    "BW1000": dict(hbm_r=1339.2e9, hbm_w=1100.5e9, mmac=475.96e12, valu_lane=80 * 64 * 1.5e9,
                   valu_pk=2.0, sfu=1916e9, cus=80, clk=1.5e9,
                   src="kra calibrate (gfx936-bw1000.json): hbm read/write, mmac_bf16, valu (pk_fma 29.7 TF), sfu_exp2"),
    "A800": dict(hbm_r=1771.2e9, hbm_w=1771.2e9, mmac=276e12, valu_lane=108 * 64 * 1.41e9,
                 valu_pk=1.0, sfu=108 * 16 * 1.41e9, cus=108, clk=1.41e9,
                 src="nv_peaks (copy 1771 GB/s, torch bf16 mm ~276 TF); FP32 / MUFU lanes from the SM spec"),
    "H20": dict(hbm_r=3821.3e9, hbm_w=3867.2e9, mmac=138.4e12, valu_lane=78 * 128 * 1.755e9,
                valu_pk=1.0, sfu=78 * 16 * 1.755e9, cus=78, clk=1.755e9,
                src="nv_peaks on h20-7 (read 3821 / copy 3867 GB/s, bf16 mm 138.4 TF); FP32 / MUFU lanes from the SM spec"),
}


def lstep_floor_bw():
    """Minimal latency of one recurrence step on gfx936 for one (head, 32-column slab) CTA of 4 waves:
    max(dependent-chain latency, MMAC issue).  chain: GEMM1 (2 dependent MMAC with 4 accumulators) ->
    v_new via LDS (write + barrier + read) -> GEMM2 (2 dependent) -> state update -> h via LDS for the
    next step.  issue: (W h + q~ h + Aqk v + k_bar^T v) MMAC instructions / 4 waves at ~8 cycles each."""
    lat = 44 * 2 + 20 + (64 + 52 + 64) + 44 * 2 + 20 + 20 + (64 + 52 + 64)
    mmac_inst = (C * D * 32 + C * D * 32 + C * C * 32 / 2 + D * C * 32) / (16 * 16 * 16)
    issue = mmac_inst / 4 * 8
    return max(lat, issue) / 1.5e9


# measured on BW1000 (bound_microbench.hip, the v8o4 dependency structure with no global memory,
# one CTA per CU): forward step (state + o) 1.23 us, state-only step (recompute / dh) 1.00 us.
# NV: no microbenchmark yet -- FLA's measured kernel per step (H20 fwd_h 1.72 us incl. loads) is an
# upper estimate, the same-structure floor is taken as ~0.8x the BW floor (range reported).
LSTEP = {"BW1000": (1.23e-6, 1.00e-6), "H20": (0.9e-6, 0.8e-6), "A800": (1.0e-6, 0.85e-6)}
# attainable MMAC for KDA's tile patterns (bound_microbench k_mmac_tile, >= 2 waves / SIMD):
# one operand register-resident 436-452 TF (0.92-0.95 of peak), both from LDS 231-237 TF (0.49)
MMAC_EFF = {"BW1000": 0.92, "H20": 0.9, "A800": 0.9}


def bounds(mname, T, H, policy="P1", lstep=None):
    m = MACHINES[mname]
    mac, valu, sfu, rb, wb, passes = inventory(policy)
    N = T * H
    flops = 2 * sum(mac.values()) * N
    lane = sum(valu.values()) * N
    tr = sum(sfu.values()) * N
    rbytes, wbytes = sum(rb.values()) * N, sum(wb.values()) * N
    t_mem = rbytes / m["hbm_r"] + wbytes / m["hbm_w"]
    t_mmac = flops / (m["mmac"] * MMAC_EFF[mname])
    # half of the elementwise lane-ops are packable fp32 pairs (fma/mul/add) where the ISA has packed math
    t_valu = (lane * (0.5 / m["valu_pk"] + 0.5)) / m["valu_lane"] + tr / m["sfu"]
    nt = T // C
    lf, lb = LSTEP[mname]
    if lstep is not None:
        lf = lb = lstep
    t_chain = nt * (lf + (passes - 1) * lb)      # forward pass (with o) + state-only passes
    # VMEM issue (TA), ta_microbench.hip on BW: row-contiguous 16 B/lane loads 21 CU-cycles per 1 KB
    # instruction (HBM working set), contiguous stores 56 per 1 KB; fragment patterns (16 rows x 32-64 B)
    # cost 64-169 per instruction.  The irreducible bytes moved with the best patterns:
    t_ta = 0.0
    if mname == "BW1000":
        t_ta = (rbytes / 1024 * 21 + wbytes / 1024 * 56) / (m["cus"] * m["clk"])
    terms = dict(mem=t_mem, mmac=t_mmac, valu=t_valu, chain=t_chain, ta=t_ta)
    thr = max(t_mem, t_mmac, t_valu, t_ta)
    return dict(terms=terms, bound=max(thr, t_chain), bracket=thr + t_chain,
                binding=max(terms, key=terms.get), flops=flops, bytes=rbytes + wbytes,
                lane=lane, sfu=tr, nt=nt, lstep=lf)


MEASURED = {   # device time (ms) of fwd+bwd, same 16-shape protocol (round tracker / op16)
    "ours": {(16384, 48): 26.43, (16384, 24): 13.09, (16384, 12): 6.75, (32768, 48): 50.13,
             (32768, 24): 25.63, (32768, 12): 12.94, (65536, 48): 98.37, (65536, 24): 48.36, (65536, 12): 24.38},
    "FLA@H20": {(16384, 48): 18.36, (16384, 24): 9.68, (16384, 12): 5.15, (32768, 48): 36.72,
                (32768, 24): 19.20, (32768, 12): 10.23, (65536, 48): 73.59, (65536, 24): 38.65, (65536, 12): 20.26},
    "FLA@A800": {(16384, 48): 26.31, (16384, 24): 13.27, (16384, 12): 7.45, (32768, 48): 52.84,
                 (32768, 24): 26.66, (32768, 12): 14.65, (65536, 48): 106.13, (65536, 24): 53.24, (65536, 12): 29.23},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="P1")
    ap.add_argument("--lstep-us", type=float, default=None, help="BW recurrence step latency (default: analytic floor)")
    a = ap.parse_args()
    mac, valu, sfu, rb, wb, passes = inventory(a.policy)
    print(f"## Work per token*head (policy {a.policy}, C={C}, d={D})\n")
    print("| | fwd | bwd | recompute | total |\n|---|---:|---:|---:|---:|")
    print(f"| MMAC kFLOP | {2*mac['fwd']/1e3:.0f} | {2*mac['bwd']/1e3:.0f} | {2*mac['recompute']/1e3:.0f} | {2*sum(mac.values())/1e3:.0f} |")
    print(f"| VALU lane-ops | {valu['fwd']:.0f} | {valu['bwd']:.0f} | {valu['recompute']:.0f} | {sum(valu.values()):.0f} |")
    print(f"| transcendentals | {sfu['fwd']:.0f} | {sfu['bwd']:.0f} | {sfu['recompute']:.0f} | {sum(sfu.values()):.0f} |")
    print(f"| HBM read B | {rb['fwd']:.0f} | {rb['bwd']:.0f} | | {sum(rb.values()):.0f} |")
    print(f"| HBM write B | {wb['fwd']:.0f} | {wb['bwd']:.0f} | | {sum(wb.values()):.0f} |")
    print(f"| sequential passes | | | | {passes} |")
    ls = a.lstep_us * 1e-6 if a.lstep_us else None
    print(f"\nrecurrence step (us, fwd / state-only): BW {LSTEP['BW1000'][0]*1e6:.2f} / {LSTEP['BW1000'][1]*1e6:.2f} measured; "
          f"analytic floor was {lstep_floor_bw()*1e6:.2f} (a model omission); NV estimates H20 {LSTEP['H20'][0]*1e6:.2f}, A800 {LSTEP['A800'][0]*1e6:.2f}"
          + (f"; using {a.lstep_us} us" if ls else "") + "\n")
    print("## Bounds (ms) and measured\n")
    print("| T | TP | machine | mem | mmac | valu+sfu | chain | ta (best patterns) | **bound (overlap)** | bracket (+chain) | binding | measured | measured / bound |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|")
    for T in (16384, 32768, 65536):
        for H in (48, 24, 12):
            for mn, key in (("BW1000", "ours"), ("H20", "FLA@H20"), ("A800", "FLA@A800")):
                b = bounds(mn, T, H, a.policy, ls if mn == "BW1000" else None)
                t = b["terms"]
                meas = MEASURED[key][(T, H)]
                print(f"| {T//1024}K | TP{96//H} | {mn} ({key}) | {t['mem']*1e3:.2f} | {t['mmac']*1e3:.2f} | {t['valu']*1e3:.2f} | "
                      f"{t['chain']*1e3:.2f} | {t['ta']*1e3:.2f} | **{b['bound']*1e3:.2f}** | {b['bracket']*1e3:.2f} | {b['binding']} | {meas:.2f} | {meas/(b['bound']*1e3):.1f}x |")


if __name__ == "__main__":
    main()
