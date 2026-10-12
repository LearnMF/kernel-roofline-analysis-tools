"""Quantitative design evaluation for the new KDA architecture, on top of kda_bound.py (2026-10-12).

Per kernel of the CURRENT production (a81ded8/44faeca), from the PMC capture (wl13): measured time,
HBM bytes and pure-VALU work per token*head.  A kernel's floor is
    max( bytes / (eta_mem * HBM),  VALU-slots / (eta_valu * VALU),  chain )
with chain = NT x the measured recurrence step (bound_microbench.hip: 1.23 us fwd / 1.00 us state at
<= 1 CTA/CU; 2.56 / 1.73 us at 192 CTAs) for the three sequential kernels.  eta = what our best
kernels already reach (l2n_apply 97% HBM; prep_a 78% VALU slots) -> 0.85 / 0.80 as "good engineering".

Designs (each row says which kernels it removes / changes and with what bytes / VALU):
  R   async-pipelined recurrences (fwd_h_o, fwd_h_bn, dhu): global tiles staged by buffer_load...lds in a
      multi-stage ring (the P-37 machinery), so a step costs ~ the microbenchmarked chain + 15% (TP8,
      latency-bound) or its bytes at eta_mem (TP2, bandwidth-bound).  A-class (bit-identical).
  B4  one fused per-chunk gradient kernel after dhu replacing s4a + s4b + bwd_tail + head_gate +
      l2n_apply (+ dav's dAqk output): reads q k v g_raw beta do + h, dh, v_new, dv2 + saved Aqk/T
      (~3.1 KB), writes dq dk dv dg_raw dbeta (~1.0 KB); VALU = today's sum of those kernels / 2
      (exp2 / gate / decay computed once instead of in 3-5 kernels).  B-class.
  C128 chunk 128: recurrence steps halve (step +40% for 2x the per-step work, est.), state natives
      halve, A-matrix tensors double.  B-class, rewrite of every kernel (not evaluated per kernel).
    python3 design_eval.py"""
HBM, VALU = 1.30e12, 7.68e12            # attainable HBM (r/w blend) B/s, VALU lane-slots/s (non-packed)
ETA_MEM, ETA_VALU = 0.85, 0.80
STEP = {"lat": (1.23e-6, 1.00e-6), "thr": (2.56e-6, 1.73e-6)}   # (fwd, state) per step

# measured per kernel: t_us at (16K,12) and (16K,48); HBM B / token*head; pure-VALU lane-ops / token*head
# (PMC valu_insts*64 minus an MMAC share; MMAC-heavy kernels marked m -- their VALU slots are a ceiling)
K = {
    "fwd_prep_a":  dict(t=(613.9, 2312.5), b=1976, v=14316, kind="par"),
    "fwd_prep_b":  dict(t=(354.4, 1470.7), b=1822, v=2967, kind="par"),
    "fwd_h_o":     dict(t=(716.6, 1766.4), b=1416, v=4762, kind="rec_fwd"),
    "l2n_apply":   dict(t=(156.0, 629.1), b=1032, v=1856, kind="par"),
    "head_gate":   dict(t=(175.0, 678.1), b=768, v=5643, kind="par"),
    "wu":          dict(t=(645.8, 2681.1), b=3260, v=9691, kind="par"),
    "fwd_h_bn":    dict(t=(537.4, 1628.2), b=1544, v=4441, kind="rec"),
    "dav":         dict(t=(278.4, 1097.9), b=1422, v=2700, kind="par"),
    "dhu":         dict(t=(673.3, 2524.5), b=2056, v=5652, kind="rec"),
    "s4a":         dict(t=(595.2, 2968.6), b=2367, v=5091, kind="par"),
    "s4b":         dict(t=(1324.6, 5942.7), b=4932, v=12465, kind="par"),
    "bwd_tail":    dict(t=(708.6, 2749.9), b=3185, v=14325, kind="par"),
}
SHAPES = {(16384, 12): 0, (16384, 48): 1}
# measured TA-busy time (us) per kernel = t x TA_TA_BUSY utilisation (PMC wl13); the TA saturates at
# 0.91-0.93, so a kernel cannot run faster than ta / 0.92 without changing its VMEM instructions
TA = {"fwd_prep_a": (258, 979), "fwd_prep_b": (302, 1336), "fwd_h_o": (299, 1281), "l2n_apply": (135, 581),
      "head_gate": (53, 222), "wu": (282, 1215), "fwd_h_bn": (170, 1196), "dav": (187, 779),
      "dhu": (351, 1494), "s4a": (301, 1770), "s4b": (924, 4322), "bwd_tail": (454, 1790)}


def floor_us(k, T, H, kind=None, b=None, v=None, ta=True):
    kind = kind or K[k]["kind"]
    b = K[k]["b"] if b is None else b
    v = K[k]["v"] if v is None else v
    N = T * H
    t_mem = b * N / (ETA_MEM * HBM)
    t_valu = v * N / (ETA_VALU * VALU)
    t = max(t_mem, t_valu)
    if kind.startswith("rec"):
        ctas = 4 * H
        lat_f, lat_s = STEP["lat"] if ctas <= 80 else STEP["thr"]
        step = (lat_f if kind == "rec_fwd" else lat_s) * 1.15
        t = max(t, (T // 64) * step)
    t *= 1e6
    if ta and k in TA and b == K[k]["b"] and v == K[k]["v"]:       # same VMEM instructions as today
        t = max(t, TA[k][SHAPES[(T, H)]] / 0.92)
    return t


def main():
    for (T, H), si in SHAPES.items():
        N = T * H
        cur = sum(K[k]["t"][si] for k in K)
        print(f"\n### {T // 1024}K / H{H} (TP{96 // H}) -- production kernels sum {cur / 1e3:.2f} ms\n")
        print("| kernel | measured us | floor us (good engineering) | measured / floor | binding |")
        print("|---|---:|---:|---:|---|")
        for k in K:
            f = floor_us(k, T, H)
            N_ = T * H
            tm = K[k]["b"] * N_ / (ETA_MEM * HBM); tv = K[k]["v"] * N_ / (ETA_VALU * VALU)
            tav = TA[k][si] / 0.92
            bind = ("TA" if tav >= f - 1 and tav > max(tm, tv) * 1e6 else
                    "chain" if K[k]["kind"].startswith("rec") and f > max(tm, tv) * 1e6 + 1 else ("mem" if tm >= tv else "valu"))
            print(f"| {k} | {K[k]['t'][si]:.0f} | {f:.0f} | {K[k]['t'][si] / f:.2f} | {bind} |")
        # designs
        rec = ["fwd_h_o", "fwd_h_bn", "dhu"]
        removed_b4 = ["s4a", "s4b", "bwd_tail", "head_gate", "l2n_apply"]
        t_R = sum(floor_us(k, T, H, ta=False) for k in rec)
        t_rec_now = sum(K[k]["t"][si] for k in rec)
        b4_v = sum(K[k]["v"] for k in removed_b4) / 2
        t_B4 = floor_us("s4b", T, H, kind="par", b=3100 + 1030, v=b4_v)
        t_b4_now = sum(K[k]["t"][si] for k in removed_b4)
        # dav keeps only dhu's operands (no dAqk fp32 output): -256 B write, -8% time
        rows = [
            ("production (measured)", cur),
            ("R  pipelined recurrences", cur - t_rec_now + t_R),
            ("B4 fused gradient kernel", cur - t_b4_now + t_B4 - 0.08 * K["dav"]["t"][si]),
            ("R + B4", cur - t_rec_now + t_R - t_b4_now + t_B4 - 0.08 * K["dav"]["t"][si]),
        ]
        h20 = {(16384, 12): 5150, (16384, 48): 18360}[(T, H)]
        a800 = {(16384, 12): 7450, (16384, 48): 26310}[(T, H)]
        print(f"\n| design | predicted us | vs production | H20 FLA / ours | A800 FLA / ours |\n|---|---:|---:|---:|---:|")
        for name, t in rows:
            print(f"| {name} | {t:.0f} | {cur / t:.2f}x | {h20 / t:.2f} | {a800 / t:.2f} |")
        print(f"\n(R: recurrences {t_rec_now:.0f} -> {t_R:.0f} us;  B4: {t_b4_now:.0f} us of kernels -> {t_B4:.0f} us, "
              f"{(3100 + 1030)} B/tok*head, VALU {b4_v:.0f} lane-ops/tok*head)")


if __name__ == "__main__":
    main()
