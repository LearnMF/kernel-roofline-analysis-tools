"""Hand-off byte table of the gfx938 production route (2631309: forward prep_a / prep_b / fwd_h, backward saved-h route).
Bytes per token-head (K = V = 128, 64-token chunks); MB at 8192/64 = B x T x H / 1e6.  Producer / consumer / sizes from
g2_runner.cpp (layoutForward / layoutBackward, forward(), backward()) and the kernels' parameter lists; checked against
the per-kernel HBM bytes of ledger.md (section 139).  python3 handoff_bytes.py [T H]"""
import sys

T, H = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (8192, 64)
TH = T * H
# name, producer -> consumer(s), write B, read B (all consumers), min W, min R, how the minimum is reached, verdict / evidence
ROWS = [
    # ---- forward
    ("Kb (k beta 2^g, bf16)", "prep_a -> prep_b", 256, 256, 0, 0, "fuse prep_a + prep_b",
     "fused CTA 0.83-0.86x (P-938ab, s135); hand-off ceiling only 1.007-1.026x fwd+bwd (s133): closed"),
    ("R (A partials, fp32)", "prep_a -> prep_b", 320, 320, 0, 0, "same fusion; bf16 R",
     "bf16 fails the FP64 bar 1.5-3.1x (s91-92); fusion closed (s135)"),
    ("QgA (q 2^g, A form)", "prep_a -> fwd_h", 256, 256, 256, 256, "-- (fwd_h reads no q / g)", "minimal"),
    ("Kn (k 2^(gl-g), B form)", "prep_a -> fwd_h", 256, 256, 256, 256, "--", "minimal"),
    ("Gn (chunk-end gate)", "prep_a -> fwd_h", 8, 8, 8, 8, "--", "small"),
    ("Wn (w, A form)", "prep_b -> fwd_h", 256, 256, 0, 0, "w = A^-1 Kb inside fwd_h",
     "option A: <= 2.6% at H = 64, 0.92-0.94x at H = 8 (s77): closed"),
    ("Un (u, C form)", "prep_b -> fwd_h", 256, 256, 0, 0, "u inside fwd_h", "same as Wn (s77)"),
    ("AqkA (Aqk copy, A form)", "prep_b -> fwd_h", 128, 128, 0, 0, "fwd_h reads the saved row-major Aqk",
     "store-deletion ceiling fwd+bwd 1.000-1.007x (s71 c2, s128): low"),
    ("Aqk (saved)", "prep_b -> dav", 128, 128, 128, 128, "--", "saved by contract"),
    ("Akk (saved)", "prep_b -> wu, s4a, s4b", 128, 384, 128, 128, "one reader (s4a + s4b merged, wu)",
     "s4 split exists for residency (s82-85): blocked"),
    ("h (saved, bf16 K x V / chunk)", "fwd_h -> wu, s4a, s4b", 512, 1536, 512, 1024, "s4a + s4b merged (wu needs h for v_new)",
     "same residency block; keep-state route (save v_new) adds retained memory: excluded"),
    # ---- backward
    ("g (fp32 chunk cumsum)", "gate -> wu (x2), s4b", 512, 1536, 0, 512, "wu / s4b recompute from g_raw (bf16)",
     "VALU scan in wu / s4b (s59: keep for s4b); wu-side alone: 671 MB; not measured -> candidate (ceiling first)"),
    ("Wb (w, B form)", "wu -> dhu", 256, 256, 256, 256, "--", "minimal"),
    ("KgA (k 2^(gl-g), A form)", "wu -> dhu", 256, 256, 256, 256, "re-use the forward's Kn",
     "retaining Kn = extra memory: excluded"),
    ("Qn (q 2^g, B form)", "wu -> dhu", 256, 256, 256, 256, "re-use the forward's QgA", "excluded (memory)"),
    ("kgB (k 2^g)", "wu -> s4a", 256, 256, 256, 256, "--", "recompute needs g (512 B): more bytes"),
    ("Vn (v_new)", "wu -> dav (C), dav -> s4b (A, in place)", 512, 512, 256, 512, "wu emits the A form, dav reads it",
     "dav prefers the C form (0.825-0.99x, s124); P-938va pays 134 MB for s4b's transpose (kept)"),
    ("dA (dAqk, fp32)", "dav -> s4b", 256, 256, 0, 0, "s4b recomputes do . v_new^T (reads both already)",
     "needs 16 KB LDS in s4b (full at 32 KB, s59 / s129-132): blocked"),
    ("dVn (dv, C form)", "dav -> dhu", 256, 256, 0, 0, "dhu computes dv = A^T do off its chain",
     "not measured -> candidate (dav + dhu, ceiling first)"),
    ("dOT (do, B form)", "dav -> dhu", 256, 256, 0, 256, "dhu reads row-major do, transposes in LDS",
     "pure re-layout of an input; not measured -> candidate"),
    ("dh (bf16 K x V / chunk)", "dhu -> s4b", 512, 512, 512, 512, "--", "recurrence state (s59)"),
    ("dv2 (A form)", "dhu -> s4a", 256, 256, 256, 256, "--", "minimal"),
    ("dwT", "s4a -> s4b", 256, 256, 0, 0, "s4a + s4b merged", "residency block"),
    ("dA (s4a, fp32)", "s4a -> s4b", 256, 256, 0, 0, "s4a + s4b merged", "residency block"),
    ("db partials", "s4a -> s4b", 16, 16, 0, 0, "s4a + s4b merged", "small"),
    ("dg (fp32)", "s4b -> bwd_tail", 512, 512, 512, 512, "bf16 dg / fold the tail into s4b",
     "bf16 fails the bar (s91-92); fold > 2x s4b (s60); layout no gain (s96)"),
]
mb = lambda b: b * TH / 1e6
print(f"### Hand-off tensors at {T}/{H} (B per token-head; MB at this shape)")
print("| tensor | producer -> consumer | W B | R B | W + R MB | min W + R MB | how the minimum is reached | verdict / evidence |")
print("|---|---|---:|---:|---:|---:|---|---|")
tw = tr = mw = 0
for n, pc, w, r, w0, r0, how, v in ROWS:
    tw += w; tr += r; mw += w0 + r0
    print(f"| {n} | {pc} | {w} | {r} | {mb(w + r):.0f} | {mb(w0 + r0):.0f} | {how} | {v} |")
print(f"| **sum** | | {tw} | {tr} | **{mb(tw + tr):.0f}** | {mb(mw):.0f} | | |")
