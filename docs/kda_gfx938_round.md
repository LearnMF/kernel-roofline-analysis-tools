# KDA on gfx938 (BW1102): round 1 (2026-10-25, nmz5, DTK 26.10 image, driver 6.4.2)

flash-train branch `feat/kda-gfx938-mk3` (local): b3d03be = e9eb1ef + gfx938 native MMAC builtins.

## 0. Port and correctness
- gfx938 lowers the generic amdgcn MMAC builtins with lit / lts set; `__builtin_hcu_mmac_*_lit_lts(..., false, false)`
  issues the gfx936 operation.  gfx936 code path unchanged.
- pytest 104 passed; 24/24 vs hip_kda P-88; 84 tensors / 11 cases bit-identical to gfx936 (bw52).
- Rule: cross-arch bitwise gates must generate inputs on the CPU.  torch's GPU RNG output depends on the
  CU count (80 vs 72), so GPU-generated inputs differ between the two cards (first gate showed 11/11 "DIFF").

## 1. Hardware ceilings (`machines/gfx938-bw1102.json`, `kra calibrate`, native MMAC builtins)
| peak | gfx936 (BW1000) | gfx938 (BW1102) | ratio |
|---|---:|---:|---:|
| CUs x clock | 80 x 1500 MHz | 72 x 1300 MHz | 0.78 |
| HBM read / write | 1339 / 1101 GB/s | 2015 / 1804 GB/s | 1.50 / 1.64 |
| L2 read | 6101 GB/s | 5116 GB/s | 0.84 |
| LDS read | 15237 GB/s | 11886 GB/s | 0.78 |
| MMAC bf16 | 476 TFLOPS | 371 TFLOPS | 0.78 |
| VALU fp32 FMA | 29.7 TFLOPS | 23.2 TFLOPS | 0.78 |
| SFU exp2 | 1916 Gop/s | 2983 Gop/s | 1.56 |
| recurrence step (1 CTA/CU, with o / state) | 1.241 / 1.015 us | 1.432 / 1.170 us | 1.15x slower |

gfx938 is memory-rich and compute-poor relative to gfx936: per-CU-clock rates are the same, there are 10% fewer
CUs at a 13% lower clock, and HBM is 1.5-1.6x faster.  The ridge point moves left by about 2x.

## 2. Ceiling cards (`results/kda-gfx938-cards-20261025/`)
| shape | total | floor | ratio (gfx936) |
|---|---:|---:|---:|
| 2048 / 16 | 1.12 ms | 0.75 | 1.49 (1.43) |
| 8192 / 64 | 13.90 | 10.75 | 1.29 (1.25) |
| 16384 / 8 | 3.85 | 2.96 | 1.30 (1.24) |

- Every kernel is issue-bound on gfx938.  wu, bwd_tail, dav and s4a were HBM-bound on gfx936.  So on this card
  KDA is already compute / issue-bound, which is the round's original goal, reached through the hardware balance.
- The instruction mix is 60-75% VALU in every hot kernel (isa_mix).  LDS ops are 4-12% and MMAC is small.  The
  issue floor is therefore set by VALU count.
- Largest gaps: wu 1.61-2.02, bwd_tail 1.30-1.87, dav 1.38-1.66, s4b 1.40-1.54.

## 3. Step 2: dispatch re-tune for 72 CUs (`sweep/sw938.jsonl`, 30 variants x 7 shapes x 2 reps)
No variant beats the default by more than 0.7%; the base rep spread is 0-0.6%.  CTA-count thresholds scaled to 72
CUs, wu orders, s4 prefetch / half-flush, kp layout, R1A / wide variants: all neutral or slower.  Turning the
P-84b/c and s4 features off costs 1-7%.  **The gfx936-tuned dispatch is already right on gfx938.**

## 4. Step 3a: gfx938-only instruction, v_cvt_pk_bf16_f32 (P-938a..d)
- `v_cvt_pk_bf16_f32` exists on gfx938 only (gfx936 assembler rejects it).  It is RNE and equals the integer RNE
  sequence for all 2^32 non-NaN fp32 inputs (NaNs come out quieted).  Throughput is 2.12 vs 4.22 cycles per wave
  conversion, and latency is 48 vs 56 cycles per dependent step.
- Builtin signature: `__builtin_hcu_cvt_pk_bf16_f32(mod0, a, mod1, b, clamp)`, where mod 1 = neg and 2 = abs.  The builtin
  beats inline asm: the asm version (P-938b) regressed up to 10% per kernel because the scheduler cannot see the
  latency.
- Bitwise: the l2norm backward in bwd_tail must keep the integer form.  With the opaque instruction, the
  compiler contracts the surrounding fp differently and K3-route dq / dk change.  All other sites are
  bit-identical (84 tensors).
- Static instruction counts fall 5-25% (wu -226, bwd_tail -1452, prep_b -478).  Non-recurrence kernels gain 1-11%.
  The recurrence kernels keep the integer form (P-938d).
- **Measurement finding:** in P-938d the 56 recurrence kernel instances are byte-identical to base, yet their
  per-kernel times moved -25%..+22% depending on shape.  An upstream kernel's changed store order changes what
  the next kernel finds in L2 (inference).  Judge only fwd+bwd totals; per-kernel deltas of unchanged kernels
  are not evidence.
- Net P-938d: 2048/16 1.000x, 4096/32 1.001x, 8192/64 1.012x, 16384/8 1.000x.  Bitwise, gfx938 only, marginal.

## 5. Step 3b: MLS (pending)
- DTK 26.10 exposes `ds_read_m*` / `ds_read_matrix_*` (with transpose), `ds_write_matrix_*` and `matrix_load_*`
  (feature `hcu-mls-insts`).  `__builtin_hcu_ds_read_m32x16_bf16` compiles to `ds_read_m32x16_b16` on gfx938
  and runs: lane l of each 16-lane group gets 8 strided elements (l, l+32, l+64, l+96, l+16, ...).
- ISATest (main-add-shaobo-saipan-b1) tags the MLS / DS-matrix tests gfx946 only, so gfx938 semantics must be
  pinned down with our own tests against the golden tables.
- Expected lever: the VALU / LDS reshuffle sequences that feed MMAC operands (ds_write_b16 / d16_hi scatter
  transposes in wu / dav / s4a, ds_bpermute in s4b / tail).

## 6. Step 3b: MLS prototype in wu (P-938m)
- Semantics pinned on BW1102 with random per-lane addresses (25600 checks, 0 mismatches): `ds_read_m32x16_b16`
  gives lane j of every 16-lane group element j % 8 of the 16 B rows addressed by source lanes
  (j / 8) + {0, 4, 8, 12, 2, 6, 10, 14}.  `_alt` is a different map: lane j gets 2 adjacent positions from 4 sources.
- wu u phase: vb is staged in natural [j][v] rows (one 8 B store per thread instead of 4 transposed b16 stores),
  and each B-fragment pair (js = 2jp, 2jp + 1) comes from one ds_read_m.  Bit-identical first try: 84 tensors, the same
  register values and the same per-accumulator MMAC order.
- wu ISA: 4303 -> 3906 instructions, including P-938d.  16 ds_read_m32x16_b16; b16 scatter stores 287 -> 223;
  b64 reads 72 -> 42.
- Time vs P-938d: wu 0.991 / 1.028 / 1.013 / 0.993x and total 0.986 / 0.991 / 0.993 / 0.991x (2048/16, 4096/32,
  8192/64, 16384/8).  Fewer instructions, no speedup: wu's 1.6-2.0x gap is stalls (2 CTAs/CU at 218-225 VGPRs),
  not issue count, so trimming the u-phase LDS traffic does not move it (inference; SQTT next).

## 7. System-level judgement (after DCU textbook 6.5.4 and the knowledge-base FA gfx938 code)
Source read in full: textbook 6.5.4 (BMZ vs NMZ FA), plus knowledge base `flash-attention-cutlass-master-dev/csrc/flash_attn_hg`
(`flash_fwd_b16_fa.h` compute_attn_mha_1rowblock_gfx938, `include/intrinsic_mls_ds.h`, `fwd/gfx938/*`, `bwd/*gfx938*`).

NMZ FA data path: HBM --`matrix_load_{32x16,32x32,64x16}_b16` (MLS: one instruction per wave tile, buffer rsrc with
row stride, optional transpose, boundary row filter in rsrc[3], async, no VGPR, no per-lane address)--> LDS
--`ds_read_matrix[_trans]_format` (scalar m0 address, MMAC operand layout)--> VGPR --`mmac_*_lit_lts(lit=1)`
(4-interleaved C that the next GEMM takes as A)--> next GEMM.  The tile grows to 128x128, mask tails are split
out, and Q stays resident in VGPR.

Our gfx936 KDA design is the BMZ answer: global row loads -> VGPR compute -> transposed LDS scatter
(`ds_write_b16` / `d16_hi`, `ds_bpermute`) -> LDS reads -> MMAC.  Inter-kernel tensors use permuted native layouts
(nat16 / pi16 / kp, A-form + B-form emits) built to make gfx936 global accesses coalesced.  Every one of those
costs VALU / LDS issue slots, and on gfx938 those slots are the binding resource (section 2), while HBM is
1.5-1.6x richer.  The gfx936 trade (spend VALU / LDS to save HBM) is inverted on gfx938.  So the gfx938 path should
be re-grown around the new data path, not patched.  Section 4 and section 6 are the patch approach, and they
confirmed it does not pay.

Direction (gfx938-only kernels; final outputs stay bit-identical to gfx936, internal layouts are free):
- D1: raw inputs (q, k, v, g, beta, do) reach MMAC through MLS -> LDS -> ds_read_matrix.  Gating products
  (x * exp2(g)) are done in-register on MMAC-layout fragments (exp2 is 1.56x faster here).
- D2: replace the gfx936 native permuted inter-kernel layouts with MLS-friendly row-major tiles.  Downstream
  kernels read them with MLS (transpose flag as needed) instead of layout-emit VALU.
- D3: register relay inside the chunk chain (A^-1 -> U = A bV -> W = A bKg -> ...): lit-interleaved C
  fragments feed the next GEMM without an LDS round trip where shapes allow.
- D4: re-run the decomposition / fusion decisions with the gfx938 four-floor model (HBM is cheap, issue is
  not).

Gate before any rewrite (P1): a microbench of the KDA chunk input stage (one chunk-head: K, G 64x128 ->
Kg = K * 2^G -> K Kg^T), gfx936-style path vs MLS-native path, on gfx938 at production occupancy.  Rewrite
kernels only if the stage shows >= 1.5x.  Order by time share: s4b 23%, prep 15%, wu 12%, tail 9-14%, dav 5%.

## 8. MLS / ds_read_matrix semantics pinned on BW1102 (examples/kda/gfx938/)
- `__builtin_hcu_matrix_load_32x32_b16(rsrc, lds(addrspace 3), 0, 0, 0, 0, 0)`: one wave instruction moves a 32x32
  bf16 tile from HBM to LDS (1024 contiguous elements, row-major).  rsrc = {addr lo, addr hi (48 bit), row stride in
  ELEMENTS, filter}.  An LDS address with +0x80000000 selects a swizzled layout (rows shifted by 8-element blocks)
  that the trans reads expect.  Async; wait with s_waitcnt.
- `ds_read_matrix_format ... element:0x2 row:0x2 col:0x1 alt:0x0` (m0 = wave-uniform LDS byte address; must be an
  SGPR value, so readfirstlane the wave-dependent address) on a plain row-major 32-wide tile: lane l gets rows
  4(l>>4)..+3 of column (l&15), and of column (l&15)+16.  That is two MMAC *B* fragments with k = row.
- `ds_read_matrix_trans_format` (same modifiers) on an MLS-swizzled tile: lane l gets row (l&15), columns
  4(l>>4)..+3 and +16.  That is two MMAC *A* fragments (js, js+1) with k = column.
- `ds_read_m32x16_bf16` (per-lane addresses): lane j of each 16-lane group gets element j%8 of the rows of source
  lanes (j/8) + {0,4,8,12,2,6,10,14}.
- An asm s_waitcnt that a separate statement emits must carry the destination registers as "+v" operands, or the
  compiler may consume them before the wait (C2/C3 first ran 60M-element mismatches).

## 9. P1 result: wu-u-phase stage microbench (`p1_mls_stage.hip`, 8192 chunk-heads, row stride 8192)
U = A(64x64) @ bf16(V(64x128) * beta), the exact rounding points of wu.  All variants are bit-identical
(67M outputs) to the gfx936 data path (A).
| variant | instrs | VALU | LDS | VGPR | full | input stage only |
|---|---:|---:|---:|---:|---:|---:|
| A gfx936 path (row loads -> VGPR scale -> transposed b16 scatter -> b64 reads) | 749 | 542 | 78 | 69 | 384.8 us | 294.1 us |
| C2 V by MLS + ds_read_matrix (B frags, k = t), scale in registers, wave owns columns | 541 | 419 | 22 | 75 | 295.2 | 214.1 |
| C4 C2 + A by MLS swizzled + trans reads (no VGPR staging, no per-lane address on the input side) | 522 | 390 | 18 | 88 | **265.1 (1.45x)** | **189.8 (1.55x)** |
| C3 persistent CTAs + double-buffered MLS | | | | | 476 | (worse: 2 CTA/CU cannot hide the serial phases) |
Gate (>= 1.5x on the input stage) passed.  The remaining full-path cost is the output side (scalar bf16
stores in the bench; in the kernels, the LDS emit plus native permuted layouts, D2).
Design rule from C2: give each wave a disjoint slice of the *scaled* operand (B columns here), so every gated
element is scaled exactly once.  The unscaled operand (A) is the one every wave re-reads.

## 10. Why FLA gains on gfx938 and we do not (kcmp_93*.jsonl, same image, same inputs)
- FLA total gfx936 -> gfx938: 1.17x at 8192/64, 1.05x at 2048/16, 1.05x at 16384/8.  flash-train: 1.01 / 1.00 / 1.00x.
- FLA's gains sit in the transposed-operand GEMM kernels: wy_dqkg 1.48-1.83x, gla_fwd_o 1.44-1.66x, recompute_w_u 1.46-1.65x
  (0.88x at 8192/64), dAv 1.58x, inter_solve 1.28-1.55x.  Our counterparts: s4b 0.92-0.96x, s4a 0.88-0.94x, wu 0.99-1.04x,
  dav 0.94-1.00x, prep_a 0.92-0.99x.  Latency / compute-bound kernels lose on both sides (FLA intra 0.74-0.90x, fwd_h / dhu
  0.76-1.20x; our gate 0.89-0.93x).
- DTK 26.10 Triton recompiles FLA for gfx938's data path (Triton cache on nmz5): 5056 `ds_read_m32x16_b16` transposed
  operand reads, all MMACs `lit`, hardware `v_cvt_pk_bf16`, autotuned warps / stages.  No `matrix_load` (HBM -> LDS MLS).
  Our hand-written kernels keep the gfx936 data path: HBM-bound ones gain from 1.5x HBM, issue-bound ones lose 0.78x
  compute, and the two cancel.
- Absolute on gfx938 we stay ahead (1.88-2.10x total), but at 2048/16 FLA's wy_dqkg (339 us) is already below our
  s4a + s4b (365 us).

## 11. SQTT on gfx938 (nmz5) works; s4b issue-time attribution
- Container needs SYS_RAWIO + SYS_PTRACE.  On nmz5 the docker wrapper rejects a rw mount of /opt/hyhal (-> /usr/local/hyhal),
  so mount it `:ro` (without hyhal the image sees no GPU).  `tanbo_kda_nmz_sqtt` runs the build from the shared
  filesystem (`/public/home/tanbo/ft938_inst`).
- `hipprof --sqtt --sqtt-type 1 --kernel-name p3_wy_s4b_kernel` gives one thread_trace json per SE (8 SEs) with issue
  time + clks per instruction; `examples/kda/gfx938/s4b_issue.py` charges each instruction its issue cycles plus the
  gap since the previous issue, by class.  The trace `offset` field is not a code PC (27655 distinct values), so the
  attribution is by opcode class.
- s4b at 2048/16 (236 VGPR -> 2 waves / SIMD), share of wave time: VMEM load issue back-pressure 19.1%, vmcnt waits 16.8%,
  barrier 10.0%, VALU int / move 10.2% + fp 8.4% + transcendental 1.9%, lgkm waits 7.5%, fp32 MMAC (16x16x8, 30 clk
  each) 6.1%, salu 4.3%, stores 3.8%, LDS write / bpermute / read 2.9 / 2.8 / 2.6%, bf16 MMAC 1.7%.  VMEM per wave
  (about 3.5 CTAs): 129 dwordx4 + 151 dwordx2 + 82 dword loads, 87 stores.
- The hypothesis "the bpermute transposes" is only 2.8%: the cost is the memory side (36%) plus barriers (10%).

## 12. s4b microbench (p1s4b.hip): load instruction form is not the bottleneck
Streaming 16 native fragment pairs per wave (64 KB per CTA, 2 CTA / CU, launch_bounds(256, 2)), fragments bit-identical:
per-lane 8 B fragment loads 1829 GB/s; MLS 2 KB -> LDS -> ds_read_b64 1920 GB/s (1.05x); per-lane 16 B pair loads
1864 GB/s.  All are near the 2.0 TB/s HBM peak in isolation.  So s4b's memory stalls are not TA width or instruction
count.  They come from latency not being hidden: 2 waves / SIMD at 236 VGPRs, loads that wait on VALU-produced
addresses / values, and barriers.  MLS's relevant property for s4b is that a prefetch into LDS costs no VGPRs, so memory
parallelism stops depending on the VGPR budget.  That is the next hypothesis to test (microbench under s4b-like
register pressure).

## 13. Register-pressure hypothesis refuted; the real s4b lever is TA work (p1rp.hip, p1s4b.hip + PMC)
- p1rp (s4b-like: phases + barriers + gating VALU, pinned live VGPRs; every variant bit-identical): per-lane loads hold
  1.67-1.72 TB/s from 63 up to 247 VGPRs (the compiler hoists the fully unrolled phases' loads).  MLS double-buffered
  prefetch is *slower*: 0.90x at 61 VGPR, 0.79x at 230 VGPR.  "MLS decouples memory parallelism from the VGPR budget"
  does not pay here.
  (Two traps on the way: launch_bounds(256, 2) capped VGPRs at 128, and a non-unrolled phase loop put the accumulator
  array in scratch, 4x slower.  Check next_free_vgpr / private_segment_fixed_size before trusting a pressure bench.)
- s4b's ceiling card on gfx938: primary resource TA (u_ta 0.64 at 2048/16, 0.71 at 8192/64); HBM only 622-754 GB/s
  (31-37% of peak); TA floor 176 / 2412 us against measured 255 / 3365 us; next floors are VALU 100 / 1595 and LDS
  75 / 1201 us.  gfx938's TA scales with CUs x clock (0.78x) while HBM grew 1.5x, so VMEM-instruction-heavy kernels
  become TA-bound.
- PMC on the streaming bench (same bytes, 8192 CTAs): TA_TA_BUSY_sum per-lane 8 B 27.0M, 16 B pairs 26.1M, 4 B 27.5M;
  **MLS 2 KB tiles 14.6M (0.54x TA work, 4x fewer VMEM instructions: 262144 vs 1048576)**.  In a non-TA-bound stream
  this shows up as only 1.05x time.  In a TA-bound kernel it is the binding resource.
- Estimate (inference): if s4b's fragment loads move to MLS, the TA floor roughly halves and s4b falls toward the VALU
  floor: s4b 255 -> about 150-180 us (2048/16) and 3365 -> about 1900-2100 us (8192/64).  That is 7-9% of fwd+bwd,
  at the 8% gate.
- Constraints: s4b already uses 32 KB of LDS per CTA x 2 CTAs / CU (the whole 64 KB), so MLS staging must reuse s4b's
  LDS phases.  The 82 narrow dword loads per wave need identifying (stream, layout).  Fragment-ordered native blocks
  can be bulk-copied by MLS without a layout change (stride 32 = contiguous 2 KB), then read with ds_read_b64.

## 14. s4b load inventory (g2_wy.cuh: p3_wy_s4b_kernel + s4b_tail; per CTA = 4 waves x 64 lanes, kt loop of 8)
Per-thread counts match the SQTT per-wave counts divided by about 3.5 CTAs per wave slot
(x4 37 / x2 43 / dword 23).  TA work of per-lane loads scales with bytes (PMC, section 13), so the table ranks streams
by bytes per CTA:
| stream | form today | bytes / CTA through TA | notes |
|---|---|---:|---|
| h + dh (native 8x8x256 blocks) | int4 per thread -> VGPR -> `stg` LDS double buffer (prefetch kt+2) | 64 KB | contiguous 4 KB tiles per kt per tensor: MLS bulk copy straight into the existing `stg` buffers, no layout change, no LDS growth, frees the pfh / pfd VGPRs and the stg ds_writes |
| gn (chunk-end g row) | ld16f per lane per kt | 32 KB | the same 512 B row read by all 64 rows of the chunk: load once (512 B) and read from LDS |
| g rows (gv) | ld16f per lane per kt | 32 KB | row-major fp32 64x128 |
| k, q rows | ld8 per lane per kt | 32 KB | row-major bf16 64x128 each |
| do / v_new fragments (ado, avn) | 8 x ld8 per lane up front | 32 KB | native fragment blocks |
| dwT tile | ld8 per thread per kt -> dwTs LDS | 16 KB | |
| dA / dAqk rows (dak, aqA) | ld16f | about 27 KB | |
| dA / dAqk column gathers (akB, aqB) | **4 B transposed gathers**, (4-A) x 8 per thread | about 20 KB | stride RA x 4 B per lane: worst coalescing; candidate: stage the 64x64 fp32 block in LDS once |
| A block | ld8 -> LDS (transposed read) | 8 KB | |
| total | | about 255 KB | stores: dq / dk / dg f4 per kt (24 dwordx4 per thread) |
Step 1 (no layout change, no LDS growth, bit-identical by construction): h / dh through MLS into `stg`, and gn from LDS.
That covers 96 KB of about 255 KB; estimated TA work -25%.  Step 2 (needs an LDS plan, since the CTA already uses
32 KB): k / q / g rows and the dA column gathers.

## 15. s4b step 1 (P-938s): bit-identical, but slower; reverted
- Change (gfx938 path only): h / dh kt tiles copied HBM -> `stg` by MLS (wave A copies 2 KB, the tile kt+2 is issued
  after the mid barrier, and `s_waitcnt vmcnt(4)` before the end barrier publishes tile kt+1); gn (chunk-end g row)
  through scalar loads (address space 4) instead of 16 B per-lane loads.  s4b VGPR 236 -> 222, LDS unchanged at 32 KB.
- Gate: 84 tensors bit-identical to gfx936, two runs.
- Time (s4b, median of 3, vs the P-938m build):
| variant | 2048/16 | 8192/64 | 16384/8 |
|---|---:|---:|---:|
| MLS h / dh only | 0.874x | 0.868x | 0.865x |
| scalar gn only | 0.961x | 0.957x | 0.947x |
| both | 0.904x | 0.894x | 0.876x |
- Reading (inference): (a) the MLS copy is issued later than the old VGPR prefetch (after the mid barrier instead of at
  the top of the iteration) and the explicit vmcnt wait puts its latency on the barrier chain, so the TA saving in
  isolation does not survive a barrier-paced loop; (b) scalar loads share lgkmcnt with LDS, so every
  `lds_barrier` (lgkmcnt(0)) now also waits for SMEM, and the 3 cndmask selects per element add VALU.
- Lesson for the methodology: a resource saving measured in isolation (TA busy -46%) is only a candidate.  Inside a
  barrier-paced kernel, the *issue position* and the *wait counter class* of the new instruction decide whether it
  shortens the critical path.  Microbenches must reproduce the barrier / wait structure, not just the bytes.
- Experiment diff kept at results/kda-gfx938-cards-20261025/patches/p938_dms_experiments.diff (P-938d cvt, P-938m wu MLS,
  P-938s s4b MLS); only P-938d / P-938m remain in the worktree, both neutral, not committed.

## 16. gfx938 global redesign, step 1 (started 2026-10-25): FLA wy_dqkg structure (fla 0.5.2 chunk_bwd.py, kept in results/)
- One CTA per (chunk, head); outer loop over K blocks (BK), inner loop over V blocks (BV); BK / BV / warps / stages
  autotuned per device (`fla_cache_autotune`, key BT / HV / STATE_V_FIRST).
- Every operand is a natural row-major tile loaded by block pointers straight from the model tensors (q, k, v, v_new,
  g, beta, A, do, dv) and from h / dh stored row-major [V, K] or [K, V].  There are no permuted native layouts.  The
  transposes (`tl.trans`, `tl.dot(b_A, ...)` with A loaded transposed) are left to the compiler, which on gfx938
  emits `ds_read_m32x16_b16` + lit MMAC (section 10).
- Work split: wy_dqkg = our s4a (dA, dv2, db) + the inter-chunk half of s4b (dq, dk, dg from h / dh / v_new / do).
  FLA's intra-chunk backward is a separate kernel (`chunk_kda_bwd_kernel_intra`), which our s4b fuses (S4).  On gfx938
  at 2048/16: FLA wy_dqkg 339 + intra 548 = 887 us against our s4a + s4b 365 us.  The fused structure is far ahead in
  total.  Only the wy_dqkg-like *part* gains from the new data path.
- Implication for the redesign: keep the fusion (fewer passes), but grow the gfx938 version's data path the FLA way.
  Natural row-major interface tensors (h, dh, v_new, do, A), operands read transposed from LDS by ds_read_m, lit
  outputs feeding the next GEMM, and autotuned tile / wave splits.  Next: the gfx938 four-floor model of the whole
  chain with natural layouts (bytes per kernel when every interface tensor is row-major, MMAC / VALU / TA floors at
  0.78x compute), then a barrier-faithful prototype of the redesigned P3 (s4a + s4b).

## 17. gfx938 four-floor model of the whole chain (examples/kda/gfx938/floors938.py, gfx938 cards)
| shape | theoretical | decomposition | fully loaded | measured | meas / full | HBM moved vs required |
|---|---:|---:|---:|---:|---:|---:|
| 2048 / 16 | 60.9 us | 388.1 | 752.1 | 1117.1 | 1.49 | 747 / 118 MB (6.3x) |
| 8192 / 64 | 974.8 | 6304.3 | 10751.3 | 13900.7 | 1.29 | 12138 / 1885 MB (6.4x) |
| 16384 / 8 | 243.7 | 1547.3 | 2958.1 | 3851.0 | 1.30 | 2976 / 471 MB (6.3x) |
- Time by primary resource: **TA 77-78% of all kernel time** (855 / 1117 us at 2048/16, 10781 / 13900 at 8192/64,
  3081 / 3851 at 16384/8), VALU the rest (prep_a, bwd_tail), HBM about 0.
- On gfx938 the chain is bound by the *per-lane VMEM work* (TA processes every lane address; it scales 0.78x while HBM
  scaled 1.5x).  It moves 6.3x the algorithm-required bytes through TA.  The fully loaded floor (cards) is 1.3-1.5x
  below measured, and the decomposition floor (HBM / MMAC only) another 1.7-1.9x below that.
- The global redesign target is therefore TA work across the whole chain: (1) fewer per-lane bytes, i.e. fewer
  interface round trips and redundant re-reads (gn-type broadcasts, wave-duplicated operand reads); (2) per-wave bulk
  loads (MLS, about half the TA work per byte, section 13) issued with enough lead that they never sit on a barrier chain
  (section 15); (3) operands read once per CTA into LDS and consumed through ds_read_m / ds_read_matrix.  Kernel
  boundaries should be chosen to minimise per-lane VMEM bytes, not HBM bytes.

## 18. TA cost per access form: first bench run invalid (to redo)
ta_forms.hip run 1 (PMC TA_TA_BUSY_sum, 4096 CTAs): k16 75.5M, k8 59.3M, k4 66.7M, broadcast 77.7M, "column" 79.6M,
MLS 264M.  Not usable: the column kernel's index folded to a contiguous pattern (VMEM_RD equals k16), and the MLS
kernel serialised on a per-iteration s_waitcnt(0) with overlapping source addresses.  Redo with explicit strided
addresses and an MLS loop without per-iteration waits.  Section 13's paired result (MLS 0.54x TA work for the same
bytes) stands.
Run 2 (fixed): 4096 CTAs, 256 MB of useful bytes per kernel (except broadcast):
| form | TA_TA_BUSY_sum | VMEM_RD | time |
|---|---:|---:|---:|
| per-lane 16 B contiguous | 74.9M | 1.05M | 833 us |
| per-lane 8 B contiguous | 59.9M | 2.10M | 654 us |
| per-lane 4 B contiguous | 67.6M | 4.19M | 746 us |
| 16 B, 16 lanes share an address (gn-like broadcast; a quarter of the useful bytes) | 77.6M | 4.19M | 849 us |
| 4 B column gather (stride 1 KB per lane) | **222.3M** | 1.05M | 2502 us |
| MLS 32x16_b16 (1 KB per wave instruction, 4 in flight) | **21.6M** | 1.05M | 431 us |
TA per byte: MLS 0.29x of per-lane 16 B, a column gather 3.0x, and a broadcast about 4x per useful byte.  Ranking of
gfx938 TA consumers to remove: column gathers (s4b akB / aqB), broadcasts (gn), then wave-duplicated reads.  The rest
should move to MLS with enough lead (section 15).
Reading of run 2: for per-lane loads TA busy is proportional to *bytes*, not instruction count (16 B: 71 cycles per
instruction, 4 B: 16), so widening loads does not help.  What helps: (a) fewer bytes through per-lane loads, (b) MLS
(0.29x per byte), (c) avoiding scattered lanes (a 1 KB lane stride costs 3x).  The s4b dA / dAqk column reads
(akB / aqB) are 16-lane-contiguous 64 B runs, so they cost about the per-lane rate, not 3x.

## 19. P3 redesign candidates on gfx938, ranked by per-lane bytes removed per CTA (s4b 255 KB, s4a ~150 KB)
1. **s4a + s4b fusion back into one P3 CTA** (split chosen on gfx936 for LDS / VGPR reasons).  It removes the dA
   round trip (s4a writes 16 KB fp32, s4b reads it twice: row dak + column akB = 32 KB), dwT (16 KB write + 16 KB
   read), the second reads of k / g / v_new / do that both kernels load, and one launch.  Estimate: 80-100 KB per CTA of
   per-lane traffic.  Cost: LDS / VGPR of the merged CTA (gfx936 P-41 / P-43 history).  To evaluate with the
   decomposition evaluator on gfx938 cards, then a prototype.
2. MLS with a real lead for the remaining contiguous streams (h, dh, do, v_new blocks): needs a third stage
   buffer (+8 KB LDS) or wave-specialised loaders.  Only after 1 settles the LDS budget.
3. Broadcast removal (gn) through LDS once per CTA, without SMEM (SMEM shares lgkmcnt with LDS, section 15).
Interface of the split today (WyS4AParams / WyS4BParams, g2_wy.cuh 804-828):
- s4a reads kgb, v_vb, A, h, dv, beta, v_row and writes dv2, dwT (bf16 64x128), dA (fp32 64x64), db_part.
- s4b reads q_row, k, A, h, do, dh, v_new, g, beta, dAqk, dwT, dA, db_part and writes dq, dk, dg, db.
- Handoff and duplicate per-lane bytes per CTA: dwT 16 KB written + 16 KB read, dA 16 KB written + 32 KB read (row
  and column), plus duplicate reads of h 32 KB, A 8 KB and beta.  That is about 120 KB of the 405 KB (s4a + s4b),
  about 30%.
- Fusion cost: keeping dA (16 KB) and dwT (16 KB) on chip needs +32 KB LDS, so residency falls from 2 to 1 CTA per CU.
  Alternative without fusion: s4a writes dA / dwT in an MLS-friendly blocked layout (it owns them), and s4b loads them
  with MLS at kernel start, where no barrier chain is waiting yet (section 15's failure mode does not apply).  This
  covers dwT + dA = 48 KB of s4b's reads at 0.29x TA.  Plan: prototype B (handoff via MLS) first (local, cheap),
  then A (fusion), evaluated with the gfx938 cards.

## 20. P-938t: dA / dAqk to LDS by MLS once per CTA in s4b (full chunks; partial varlen chunks keep per-lane loads)
- 16 MLS tiles per CTA (wave A: tiles 2A, 2A+1 of each tensor), one vmcnt(0) + barrier at kernel start, rows and
  column gathers then read from LDS (tile-blocked address), second barrier before the h / dh staging reuses LDS.
- Gate: 84 tensors bit-identical, two runs.  s4b global_load 114 -> 78, VGPR 237, LDS 32 KB.
- s4b: 2048/16 256.1 -> 247.6 us (1.034x), 4096/32 1.028x, 8192/64 1.026x, 16384/8 0.993x.  Total fwd+bwd within
  +-1% (1.011 / 0.994 / 0.997 / 0.989).  Removing about 20% of s4b's per-lane bytes buys about 3%: s4b is not purely
  TA-throughput-bound (TA utilisation 0.64-0.71); the rest is latency / dependency structure at 2 waves per SIMD.
- Kept as candidate (neutral overall), not committed.

## 21. Working plan (autonomous run, 2026-10-25 night)
Consistent picture so far: TA-backpressure stalls (SQTT: VMEM issue stalls 19% of s4b wave time) scale with per-lane
bytes.  P-938t removed about 20% of s4b's per-lane bytes and gained about 3%, matching 19% x 20% x (part of the vmcnt
waits).  So the gfx938 lever is *per-lane bytes across all TA-primary kernels* (s4b, s4a, dhu, fwd_h, wu = 78% of
time), each converted to MLS or removed, with MLS issued far enough ahead (P-938s lesson).  Expected chain gain if
every TA-primary kernel loses most per-lane bytes: about 10-15% (inference).
Next items, in order:
1. s4b gn broadcast (32 KB per-lane per CTA for 512 B useful): wave 3 lanes m = 15 already hold g[63] = gn in gv.
   Publish it through LDS once per kt (needs 128 B of LDS; find room, e.g. gM / cs reuse) instead of the 16 B
   per-lane load.  It removes 12.5% of s4b's per-lane bytes with no SMEM (lgkmcnt) side effect.
2. s4b h / dh: MLS with lead.  Restructure the stg ring so tile kt+2's MLS is issued at the top of iteration kt
   (needs a free buffer at that time: a 3-stage ring is +8 KB, so first find LDS: TA / TQ / TK tables 13 KB).
3. Same ledger for dhu, fwd_h, s4a, wu (cards: TA primary), per-lane bytes per CTA, then the cheapest MLS
   conversions (contiguous blocked streams first).

## 22. P-938u: gn published by wave 3 through the TK pad column (on top of P-938t): no gain, reverted
- Bit-identical (two runs), VGPR 242, s4b global_load 76.  s4b against the base: 2048/16 1.014x (P-938t alone 1.039x),
  4096/32 0.995x, 8192/64 1.000x, 16384/8 0.976x.  It is worse than P-938t alone: the four LDS reads now sit in front
  of pass K (an lgkmcnt wait on the critical path), while the removed broadcast load hit the TCP cache and was off the
  critical path.
- Conclusion from P-938s / t / u: in s4b, removing per-lane bytes gives at most about 3%.  s4b's time is set by its
  dependency / barrier critical path at 2 waves per SIMD, not by TA throughput (utilisation 0.64-0.71).  The
  bytes-based TA ranking of section 19 is an upper bound, not a predictor.  The next lever must shorten the critical
  path itself: more waves in flight (occupancy / CTA geometry) or fewer barrier-separated phases per kt.

## 23. Next lever: s4b critical path / occupancy (plan)
- s4b: 4 waves per CTA, 237 VGPR, so at most 2 waves per SIMD (512 / 237) whatever the LDS.  With 32 KB LDS that is
  2 CTAs per CU = 8 waves per CU.  Per kt it runs about 3 barriers (mid, table WAR, end): 24 barriers per CTA.
- Long-lived VGPR holders across the kt loop: ado[8] + avn[8] (16 v4bh = 32 VGPR), dak[4] (16), aqA / aqB / akB
  (about 20), aa[4] (8), plus the accumulators.  About 76 VGPRs are loop-invariant operands that could live in LDS if
  LDS were free.  With P-938t, dA / dAqk are no longer needed in registers if read from LDS per kt (but that LDS is
  reused by the loop tables / stg).
- Probe first (cheap, decisive): measure s4b time against forced residency on gfx938 (LDS top-up probe of
  results/kda-decomp-calib, downwards) and the barrier share by SQTT.  If going from 2 to 1 waves per SIMD costs a
  lot more than linear, then 2 -> 3 is worth a VGPR <= 168 redesign (operands to LDS, 8-wave CTA).

## 24. s4b residency probe (gfx938, P-938t build; +24 KB LDS pad -> 1 CTA per CU)
| shape | 2 CTA / CU | 1 CTA / CU | ratio |
|---|---:|---:|---:|
| 2048/16 | 242.6 us | 377.2 | 1.555 |
| 8192/64 | 3349.8 | 5111.1 | 1.526 |
| 16384/8 | 835.6 | 1307.0 | 1.564 |
Halving the waves in flight costs 1.53-1.56x, not 2x: going from 1 to 2 CTAs per CU recovers 78% of ideal.
Extrapolating (inference): 2 -> 3 CTAs per CU (or 8-wave CTAs at the same per-CU LDS) would give s4b about
1.2-1.3x until TA caps it (utilisation 0.64-0.71 now, about 0.95 at 1.4x).  That is about 4-6% of total fwd+bwd from
s4b alone.  It needs VGPR <= 168 (3 waves per SIMD) and LDS <= 21 KB per CTA, or an 8-wave CTA with VGPR <= 128.
That is a structural rewrite of s4b, and it is the largest single lever found on gfx938 so far.

## 25. Residency survey (gfx938 cards, 8192/64; share of kernel time, VGPR, resident CTAs per CU)
s4b 24.2% (236, 2) | wu 11.7% (220, 2) | dhu 10.8% (124, 1; recurrence grid of 256 CTAs) | s4a 10.2% (216, 2) |
prep_a 9.9% (152, 3) | bwd_tail 9.0% (112, 3) | fwd_h 8.6% (192, 2; 256 CTAs) | prep_b 6.5% (72, 3) | dav 5.2% (76, 3) |
gate 3.4% (92, 3).
- Kernels at low VGPR but residency 3 (prep_b, dav, bwd_tail, gate: 21.1% of time together) are limited by LDS or by
  launch bounds, not VGPR.  Next: read each kernel's group_segment_fixed_size and launch_bounds from the gfx938 ISA.
  An LDS trim to <= 16 KB (4 CTAs) is a cheap occupancy step if their time follows residency like s4b (section 24).
- The 1.55x residency sensitivity of s4b suggests the same for the other issue / latency-bound kernels on gfx938.
  Occupancy is therefore the first chain-wide lever to sweep, ahead of per-lane byte removal.
LDS / VGPR per kernel from the gfx938 ISA (P-938t build): fwd_intra2a (prep_a) 20480 B / 147; fwd_intra2b (prep_b)
**17408** / 70-77; dav **17408** / 74; bwd_tail 6144 / 114; dhu_v8b 20992 / 123-127; fwd_h_v8o4 12800-14848 /
122-197.  prep_b and dav are 1 KB over the 16 KB line: 64 KB / 17 KB = 3 CTAs per CU, and at 16 KB they would get 4
(VGPR < 128 allows 4).  Together they are 11.7% of the time at 8192/64.  This is the cheapest occupancy step: find
the 1 KB (padding / staging), trim it bit-identically, and measure.

## 26. P-938v: dav at 16 KB LDS (4 CTAs per CU), XOR swizzle instead of padding
- Unpadded without swizzle: bit-identical but dav 3.4-4x slower (bank conflicts).  With an XOR of column bits 2..5
  by the row's low 4 bits (vectors stay contiguous, 16-row operand reads conflict-free): LDS 16384, VGPR 59,
  bit-identical.
- dav: 2048/16 59.3 -> 54.9 us (1.080x), 16384/8 189.4 -> 175.5 (1.079x), but 8192/64 708.0 -> 809.7 (0.874x).
  Total: 1.003x / 1.012x / 0.998x.
- More residency helps small and medium grids and hurts the big one (8192 CTAs: more waves contend for TA / L2).
  Next: make residency a launch knob (dynamic LDS top-up to force 3 CTAs per CU on big grids), and check the
  crossover with the 24-shape sweep.
Residency knob sweep (dynamic LDS on top of the swizzled 16 KB slab; ex0 = 4 CTAs per CU, ex1024 / ex5120 = 3):
| shape | 4 CTA | 3 CTA (+1 KB) | 3 CTA (+5 KB) |
|---|---:|---:|---:|
| 2048/16 | 55.1 us | 61.3 | 61.4 |
| 4096/32 | 173.7 | 194.3 | 192.7 |
| 8192/32 | 336.2 | 370.2 | 371.5 |
| 8192/64 | 689.2 | 762.7 | 731.5 |
| 16384/8 | 176.7 | 194.1 | 195.6 |
| 16384/64 | 1390.5 | 1440.5 | 1443.9 |
| 32768/16 | 654.0 | 723.8 | 723.0 |
At 4 CTAs per CU dav is faster on every shape (1.04-1.11x).  The 8192/64 regression in the first A/B was a noisy
sample (3 shapes in parallel on separate GPUs); the paired sweep on one GPU is consistent.  P-938v kept.
Next: the same treatment for fwd_intra2b (prep_b, s_post T2 [128][68] -> [128][64] swizzled, 17408 -> 16384 B) as
P-938w.

## 27. P-938w: prep_b (g2::r4::g2_fwd_prep_b_kernel, the production prep_b) at 16 KB LDS, same swizzle as P-938v
(First attempt edited fis::g2_fwd_intra2b_kernel, which is not on the production route: reverted.  Lesson: confirm
the production kernel symbol from the profiler names before editing.)
- LDS 17408 -> 16384, VGPR 110, bit-identical (two runs, with stale gate outputs deleted first).
- prep_b: 2048/16 73.9 -> 69.6 us (1.062x), 4096/32 1.025x, 8192/64 0.967x, 16384/8 1.033x.  Total with P-938t + v + w:
  1.009x / 1.006x / 0.998x / 1.006x against P-938t alone.
- Running tally on gfx938 (bit-identical, uncommitted): P-938t (s4b dA / dAqk by MLS), P-938v (dav 4 CTAs),
  P-938w (prep_b 4 CTAs): about +1% total.  Each is small.  The large lever stays the 2-CTA kernels s4b / s4a / wu (46%
  of time, VGPR 205-240, LDS 27-32 KB), where the residency probe (section 24) predicts about 1.15-1.25x each at 3 CTAs.
  That needs VGPR <= 168 and LDS <= 21.3 KB per CTA.

## 28. Measurement hygiene notes (this round)
- `g2_env_long` / `g2_getenv` read the environment only with FTRAIN_KDA_DEV=1.  Every probe knob must run with it,
  otherwise the probe silently measures the default.  The first residency probe run hit this, and its gate / tail
  "differences" (up to 9% on gate at 8192/64) were pure run-to-run noise.  So single-kernel deltas below about 5%
  from 3 reps, with 3 GPUs running at once, are not significant for small kernels.
- Gate scripts must delete earlier gate outputs before running.  A failed build otherwise "passes" on stale JSON.

## 29. Residency sensitivity of every big kernel on gfx938 (dynamic LDS top-up, FTRAIN_KDA_DEV=1, kernel us)
| kernel (resident now) | probe | 2048/16 | 8192/64 | 16384/8 |
|---|---|---:|---:|---:|
| wu (2) | 1 CTA / CU | 140.1 -> 195.5 (1.40x) | 1615.6 -> 2552.9 (1.58x) | 434.0 -> 651.8 (1.50x) |
| s4a (2) | 1 CTA / CU | 110.4 -> 158.8 (1.44x) | 1388.6 -> 2019.5 (1.45x) | 272.3 -> 465.9 (1.71x) |
| s4b (2) | 1 CTA / CU (section 24) | 1.555x | 1.526x | 1.564x |
| bwd_tail (4) | 3 CTA / CU | 98.0 -> 127.4 (1.30x) | 1264.3 -> 1471.9 (1.16x) | 280.5 -> 363.8 (1.30x) |
| bwd_tail (4) | 2 CTA / CU | 152.9 (1.56x) | 1881.2 (1.49x) | 485.0 (1.73x) |
| gate (5) | 4 CTA / CU | 36.8 -> 39.8 (1.08x) | 471.4 -> 524.6 (1.11x) | 121.5 -> 124.8 (1.03x) |
| gate (5) | 2 CTA / CU | 45.3 (1.23x) | 507.6 (1.08x) | 133.7 (1.10x) |
Every big kernel on gfx938 is latency-bound at its current residency: one CTA less costs 16-71%.  The chain-wide
gfx938 lever is therefore **residency**, ranked by (share x sensitivity): s4b, wu, s4a (2 -> 3 CTAs: VGPR <= 168
and LDS <= 21.8 KB) and bwd_tail (4 -> 5: VGPR <= 102).  prep_a (3, VGPR 139-163, LDS 21.5 KB) is not measured
yet (profiler key mismatch).

## 30. P-938x: bwd_tail to 5 waves per SIMD by a VGPR cap: no gain, reverted
- `amdgpu_waves_per_eu(5)`: the compiler went to 48 VGPR with 304-348 B of scratch, and tail was 0.58-0.65x.
- `amdgpu_num_vgpr(102)`: 102 VGPR with 36 B of scratch, tail 1.036x / 0.971x / 0.997x / 0.976x (2048/16, 4096/32,
  8192/64, 16384/8).  The extra wave is eaten by the spills.
- Lesson: residency bought by a register cap only pays when the cap is nearly free.  The probe measures the value of
  residency at a fixed instruction stream; a cap changes the stream (spill traffic goes through TA as well).

## 31. P-938y: wu with the A fragments in VGPRs (no sA, LDS 18176 B) [+ VGPR cap 168]: not bit-identical, reverted
- With the cap: 168 VGPR, 108-116 B of scratch.  The gate fails on the K3 cases (in-kernel l2norm: dq / dk of
  k3_d_8192_48 and k3_vl_5000_16), with and without the cap.  The K3 cases take the no-saved-h (p1) route, where wu
  also emits Wn / Un / Kn.  That branch differs under the change and needs a dedicated look before any retry.  The
  spill size also says a 3-CTA wu needs a real VGPR restructure, not a cap.
- Also reverted: P-938m (wu u-phase MLS, neutral), so the gfx938 delta in the worktree is P-938d + t + v + w.

## 32. CTA-order knobs re-swept on gfx938 (6 shapes, 3 reps): defaults stay
- wu order (0 / 1 / 2 / 3 with m = 2, 4, 8): best alternative within +-1.8% of the default everywhere (noise level).
- s4 head-fastest off (KDA_G2_S4_HF=0): s4a 0.66-0.95x, s4b 0.84-0.98x.  Head-fastest is essential on gfx938 too.

## 33. Small-H recurrences on gfx938: the grid does not fill 72 CUs
- 16384/8: fwd_h (32 CTAs = 8 heads x 4 V-slabs of 32 columns) 531 us + dhu (32 CTAs) 543 us = 28% of the total,
  with 40 of 72 CUs idle for the whole recurrence.
- Slab-width scaling (dev knobs; kp off for both, so they are comparable), fwd_h us:
| shape | 32-col (4 CTAs / head) | 64-col (2 CTAs / head) | ratio | default (kp on, 32-col) |
|---|---:|---:|---:|---:|
| 4096/8 | 172.9 | 383.9 | 2.22x | 142.0 |
| 8192/16 | 380.1 | 758.9 | 2.00x | 290.9 |
| 16384/8 | 661.9 | 1503.8 | 2.27x | 527.8 |
| 32768/8 | 1315.6 | 2998.0 | 2.28x | 1045.6 |
  The step latency scales with the slab width (per-CTA V work), not with the shared K-side loads.  So 16-column slabs
  (8 CTAs per head, 64 CTAs at H = 8) should roughly halve fwd_h where the grid is far below 72 CUs.
- P-938z (dev): g2_fwd_h_v8o4w_kernel<NVH = 1> (one 16-column V-tile, a single 8 B U load per lane per step).  It
  is the wide kernel's generic body.  Being measured: forced with kp off against the default (kp on, 32-col).

## 34. Bisection: the K3 dq / dk mismatch was a broken P-938d, not P-938t / y / z
- Symptom: after reverting P-938x / P-938y by `git checkout <file>`, the gate failed on the K3 cases (in-kernel
  l2norm), dq / dk only: about 400 of 50M elements, 1 bf16 ulp, scattered over all chunks and heads.
- Bisection from the passing HEAD (b3d03be): HEAD + d (full P-938d) passes, HEAD + v + w passes, HEAD + d + v + w
  passes, HEAD + d + t + v + w + z passes.  HEAD + d without its g2_bwd_tail.cuh part fails, and so does HEAD + t
  alone (t without d).
- Cause: `git checkout g2_bwd_tail.cuh` (reverting P-938x) also dropped P-938d's two-line change there:
  `bf_rne` -> `bf_rne_int` in the l2norm-backward output.  On gfx938 `bf_rne` goes through the hardware convert, and
  the compiler's FMA contraction of the expression feeding it then depends on surrounding code.  The integer RNE pins
  the gfx936 arithmetic.  The K3 route's l2norm backward is where 1 ulp of a contraction difference shows.
- Rules: (1) revert an experiment with its own reverse patch, never by checking out a file that holds several
  experiments; (2) on gfx938 use the integer RNE (bf_rne_int) wherever the convert's operand is an arithmetic
  expression, so contraction cannot depend on the consumer; (3) the K3 cases are the sensitive part of the gate
  and must stay in it.
- Consistent gfx938 set now: P-938d + t (with a 256 B alignment guard) + v + w (+ z, dev-only launch path).
  Bit-identical.

## 35. P-938z: 16-column fwd_h slabs on gfx938 at H <= 8 (production rule)
- g2_fwd_h_v8o4w_kernel<NVH = 1>: the wide kernel's generic body with one 16-column V-tile (U loaded as the even / odd
  8 B half of its pair's 16 B lane quad), 8 CTAs per head.  It needs the plain Kn / Gn layout (kp off).
- Measured, forced with kp off vs the default (kp on, 32-col), fwd_h / total:
| shape | 16-col fwd_h | total |
|---|---:|---:|
| 4096/8 | 141.9 -> 126.7 us (1.12x) | 1.014x |
| 16384/8 | 530.9 -> 450.5 (1.18x) | 1.016x |
| 32768/8 | 1046.8 -> 890.2 (1.18x) | 1.023x |
| 2048/16 | 0.87x | 0.993x |
| 8192/16 | 0.74x | 0.968x |
| 16384/32 | 0.64x | 0.961x |
- Rule (one host rule for every kp producer / consumer: g2_kp_layout in g2_common.h and g2KpLayout in the runner):
  on gfx938 at H <= 8 use kp off and the 16-col fwd_h; everything else unchanged (gfx936 untouched).
  Bit-identical (two gate runs).
- Why it works: at H = 8 the default grid is 32 CTAs on 72 CUs, and the fwd_h step latency scales with the per-CTA
  V width (section 33), so halving the width with twice the CTAs cuts the chain time.  At H >= 16 the grid already
  fills the CUs, and the duplicated K-side loads cost more than the shorter step saves.
- dhu (543 us at 16384/8, 32 CTAs of 32 columns, 8 waves) has no width template.  A 16-column dhu is the analogous
  next step at small H.

## 36. Methodology distilled from the gfx938 round (so far)
1. **Model first, then the residency probe, then the code.**  The four floors (section 17) said TA, but the probes
   (sections 24 and 29) showed every big kernel is latency-bound at its residency: one CTA less costs 16-71%.
   Bytes-based ceilings are upper bounds.  The dynamic-LDS residency probe is the cheapest decisive measurement
   (no code change: dynamic LDS on the launch) and should run before any rewrite.
2. **A resource saving only counts if it lands on the critical path.**  MLS saves 46-71% TA work per byte in
   isolation.  Inside a barrier-paced kernel it lost 10% (issue position, vmcnt class: P-938s) or gained 3% (P-938t).
   Removing a broadcast load (P-938u) lost: it was off the critical path.
3. **Residency is bought with layout, not caps.**  A swizzle in place of padding (dav P-938v, prep_b P-938w) gave
   +1 CTA per CU for free, bit-identically.  Register caps (P-938x: 102 VGPR; P-938y: 168) spilled and lost.
4. **Fill the machine at small grids.**  On 72 CUs, H = 8 recurrences launch 32 CTAs.  The step latency scales with
   the per-CTA width, so narrower slabs and more CTAs win (P-938z, fwd_h 1.12-1.18x).  The same rule loses once
   the grid fills the CUs.
5. **Hygiene.**  Dev knobs need FTRAIN_KDA_DEV=1.  Gate scripts must delete stale outputs.  Revert experiments with
   their own reverse patches.  The K3 (in-kernel l2norm) cases are the sensitive part of the bit-exact gate.
   Single-kernel deltas below about 5% from 3 reps on parallel GPUs are noise for small kernels.

## 37. Correction (2026-10-26): which instructions are really gfx938-only
Checked with the DTK 26.10 compiler (`__has_builtin` per --offload-arch) and by counting opcodes in the FLA Triton
caches (620 KDA-related kernels per machine) and in our gfx936 / gfx938 builds (examples: insn_probe.sh):
| item | gfx936 compiler | FLA on gfx936 | FLA on gfx938 | flash-train gfx936 / gfx938 |
|---|---|---|---|---|
| ds_read_m32x16_b16 | available | 4936 | 5056 | 0 / 0 |
| MMAC `lit` modifier | no builtin | 0 of 34676 | 35156 of 35156 | 0 / 0 |
| v_cvt_pk_bf16_f32 | no builtin | 0 | 22128 | 0 / 12126 (P-938d) |
| matrix_load (MLS) | no builtin | 0 | 0 | 0 / 32 (P-938t) |
| ds_write / ds_bpermute | -- | 33700 / 6200 | 21632 / 11792 | 12294 / 4458 (both) |
- Section 10's sentence was wrong on ds_read_m: it is a gfx936 instruction too, and FLA already uses it on gfx936.
  It is not why FLA got faster on gfx938.  It is an instruction our hand-written kernels never use on either
  architecture.
- Really gfx938-only: the MMAC `lit` modifier (semantics and gain not measured in isolation yet) and the hardware
  bf16 convert (measured 0-1.2% for us, P-938d).  MLS is gfx938-only, and FLA does not use it.
- What does differ in FLA's gfx938 code: the autotuned tiling (ds_write -36%, ds_bpermute +90% in the same kernels)
  and `lit` on every MMAC.  Together with the 1.5x HBM, which helps FLA's more memory-bound kernels, these are the
  candidate causes of FLA's 1.05-1.17x.  The split between them is not measured.
Method note and assembler check (examples/kda/gfx938/insn_probe.sh, asm_probe.sh):
- `__has_builtin` only shows what the compiler front end exposes.  The Triton / ISA counts are static occurrences:
  every cached autotune candidate is counted, and the name filter matched all 620 cache entries.  They show that an
  instruction is used, not its runtime share.  insn_probe.sh's `mmac_lit` pattern looks for `lit` in the opcode name
  and therefore missed the trailing `lit` modifier (it printed 0 for FLA on gfx938).  The lit counts in the table come
  from a separate grep of `v_mmac... lit`.
- The ISA itself, checked with DTK's llvm-mc (-mcpu=gfx936 / gfx938):
  ds_read_m32x16_b16: accepted on both, same encoding.
  v_mmac_f32_16x16x16_bf16 ... lit: gfx936 rejects ("The lit and lts field not support on this GPU"); gfx938 accepts.
  v_cvt_pk_bf16_f32: gfx936 rejects ("instruction not supported on this GPU"); gfx938 accepts.
  matrix_load_32x32_b16 (MLS): gfx936 rejects; gfx938 accepts.

## 38. Whole-chain diagnosis on gfx938 (a83ac15; SQTT of all 11 production kernels at 2048/16 and 8192/64)
Tools: cap_chain.sh (capture), chain_mix.py (dynamic issue + stall per class, VALU split into math vs data movement),
lds_inv.py (static LDS op inventory).  Data: results/kda-gfx938-chain-20261026/.  Weights = measured kernel times.
Chain shares (time-weighted; the two shapes agree within 1-2 points):
| group | instructions | cycles (issue + stall) |
|---|---:|---:|
| real compute (MMAC + fp VALU + transcendental) | 29% | 13-15% |
| VALU data movement (int / address 12%, logic / shift 11%, select / move 10%, cvt / pack 7%) | 40.5% | 17% |
| LDS (read / write / bpermute) | 10-11% | 8% |
| VMEM issue (incl. TA back-pressure) | 4% | 24% |
| waits (vmcnt 15-16%, barrier 8-9%, lgkmcnt 7%) | 7% | 29-31% |
| SALU | 9% | 6% |
- Only 13-15% of the cycles do math.  Moving data costs more VALU issue (17%) than computing does, and memory issue
  plus waits take 55%.  With 0.78x VALU / LDS / TA on gfx938, the data-movement VALU work is exactly where the
  compute cut hurts, while HBM sits at about a third of its peak.
- Per-kernel signatures (cycles %, 8192/64; 2048/16 similar):
| kernel (time share) | compute | VALU move | LDS | VMEM | wait_vm | barrier | move / math instr | static b16 transposed LDS writes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| s4b (24%) | 19 | 12 | 9 | 22 | 16 | 9 | 0.9 | 0 (84 bpermute) |
| wu (12-13%) | 8 | 27 | 11 | 13 | 21 | 7 | 5.6 | 287 |
| s4a (10-11%) | 7 | 11 | 10 | 43 | 5 | 6 | 4.1 | 80 |
| prep_a (10-11%) | 34 | 21 | 4 | 9 | 9 | 9 | 0.7 | 32 |
| bwd_tail (9-10%) | 12 | 11 | 1 | 13 | 56 | 4 | 0.7 | 0 |
| dhu (8-11%) | 8 | 14 | 11 | 34 | 3 | 20 | 8.6 | 20 |
| fwd_h (8-9%) | 11 | 17 | 14 | 31 | 2 | 7 | 6.5 | 8 |
| prep_b (6.5-7%) | 3 | 10 | 6 | 50 | 12 | 13 | 8.2 | 0 |
| dav (5%) | 2 | 19 | 10 | 24 | 28 | 7 | 55 | 96 |
| gate (3.5%) | 15 | 67 | 0 | 8 | 3 | 3 | 0.5 | 0 |
- Lever map: (1) layout transforms in the load path (ds_read_m32x16, which gfx936 has too, plus gfx938's
  ds_read_matrix) for the move-heavy kernels with transposed b16 LDS staging: wu, dav, s4a, dhu / fwd_h.
  (2) MLS bulk loads issued with lead for the VMEM-issue-heavy kernels with blocked contiguous streams: s4a, prep_b,
  dhu, fwd_h, s4b.  (3) Residency (VGPR / LDS freed by 1 and 2) for the latency-bound ones: s4b, wu, s4a.
  (4) More memory parallelism for bwd_tail (56% vmcnt wait).
- ds_read_m32x16_b16 semantics are identical on gfx936 and gfx938: the same random-address probe (dsm16.hip, 64
  blocks) gives byte-identical output on BW1000 and BW1102.  Lane j of each 16-lane group receives element j % 8 of
  the 16 B granules addressed by source lanes j / 8 + {0, 4, 8, 12, 2, 6, 10, 14}.  So lever (1) applies to both
  architectures with one code path.

## 39. wu, step by step (gfx938 measured on nmz5; gfx936 pending a quiet machine)
- P-938m (u phase) re-enabled on both architectures (guard = `__has_builtin(ds_read_m32x16)`) + P-938m2 (v_new's
  saved-h B tiles in a natural [k][v] layout, read back as B-fragment pairs by ds_read_m, granule XOR swizzle):
  bit-identical on gfx936 (bw52) and gfx938.  ISA (gfx938 wu<false,true,true>): ds_write_b16 287 -> 159,
  ds_read_m 0 -> 48, VGPR 231 -> 213.  Time: wu unchanged (gfx938 0.98-1.00x; gfx936 on bw52 GPU0, sequential:
  144 vs 145 us).  The removed LDS writes were off the critical path (wu: 25% of cycles in waits, 2 waves per SIMD).
- P-938r (residency): A fragments in VGPRs (no sA: LDS 26880 -> 18176 B) and no exp2f(g) / kg carried to phase 3
  (phase 3 re-reads k / g; same expressions).  VGPR 213 -> 197 without a cap.  With amdgpu_num_vgpr(168): 3 CTAs
  per CU, scratch 24 B on gfx936 / 100 B on gfx938.  Bit-identical on gfx938 (two runs each).
  First A/B on gfx938: cap 168 -> wu 0.87-0.90x (spills); no cap -> wu 1.06x at 2048/16 and 4096/32 (other shapes
  noisy: nmz5 is shared, other users' unit tests run on it).  A 5-rep interleaved min / median A/B is running.
- Measurement environment: bw61 (all 8 GPUs busy with another user's Megatron run since 11:41) and bw52 (22-57%
  busy) cannot give clean gfx936 numbers today, and other bw hosts are not accessible.  nmz5 is shared, so A/B now
  uses 5 interleaved reps on one GPU and reports min and median.
- 5 interleaved reps (min / median), wu<false,true,true>:
| variant | gfx938 (nmz5) wu | gfx938 total | gfx936 (bw52) wu | gfx936 total |
|---|---:|---:|---:|---:|
| P-938m + m2 + r, no cap | **1.06-1.10x** | 1.003-1.009x | 0.86-0.95x | 0.98-0.99x |
| + VGPR cap 168 (3 CTAs / CU) | 0.88-0.90x | 0.97-0.99x | 0.82-0.88x | 0.97-0.98x |
- Reading: trading bandwidth for registers (phase 3 re-reads g instead of carrying exp2f(g)) pays on gfx938 and
  loses on bandwidth-tight gfx936.  The ds_read_m layouts alone are neutral on gfx936.  Forcing a third CTA with a
  register cap spills and loses on both.  So wu's 3-CTA residency needs a real live-range restructure (the v_new
  phase holds we8 + wA = 32 VGPRs plus the accumulators), not a cap.
- Kept, gfx938 only (G2_WU_MLS / G2_WU_LEAN under __gfx938__).  The gfx936 device code is verified identical to
  a83ac15: 127 kernels, 0 function bodies and 0 kernel descriptors differ.  gfx938 gate 84/84 bit-identical.
- 5 interleaved reps (min / median), wu<false,true,true>:
| variant | gfx938 (nmz5) wu | gfx938 total | gfx936 (bw52) wu | gfx936 total |
|---|---:|---:|---:|---:|
| P-938m + m2 + r, no cap | **1.06-1.10x** | 1.003-1.009x | 0.86-0.95x | 0.98-0.99x |
| + VGPR cap 168 (3 CTAs / CU) | 0.88-0.90x | 0.97-0.99x | 0.82-0.88x | 0.97-0.98x |
- Reading: trading bandwidth for registers (phase 3 re-reads g instead of carrying exp2f(g)) pays on gfx938 and
  loses on bandwidth-tight gfx936.  The ds_read_m layouts alone are neutral on gfx936.  Forcing a third CTA with a
  register cap spills and loses on both.  So wu's 3-CTA residency needs a real live-range restructure (the v_new
  phase holds we8 + wA = 32 VGPRs plus the accumulators), not a cap.
- Kept, gfx938 only (G2_WU_MLS / G2_WU_LEAN under __gfx938__).  The gfx936 device code is verified identical to
  a83ac15: 127 kernels, 0 function bodies and 0 kernel descriptors differ.  gfx938 gate 84/84 bit-identical.

## 40. wu structural fix: 3 CTAs per CU without spills (P-938k, gfx938)
- Where wu's VGPR peak was (compile-time cuts of g2_wu_kernel<false,true,true> on gfx938): phase 1 alone 102,
  + w-MMAC / emits 115, + fill2 124, + u-MMAC / v_new / Un emit 126, full kernel 197.  With the phase-3 Kn branch
  compiled out: 173.  With the phase-3 QS body removed: 126.  So the peak was phase 3, not v_new.  Splitting the QS
  loads into two halves did not help (175); `#pragma unroll 1` on the halves did: 128, no scratch.  The full unroll
  let the compiler hoist all of phase 3's addresses / loads at once.
- Changes (gfx938 only): template KN = 0 compiles the Kn branch out; the launcher uses it when Kn == nullptr (the
  saved-h / keep-state routes) on gfx938.  Phase 3's QS stores go in two non-unrolled halves of 4 quads.  Result:
  wu VGPR 128, LDS 18176, scratch 0 -> 3 CTAs per CU (4 waves per SIMD by VGPR; LDS allows 3).
- Gate 84/84 bit-identical (two runs).  5 interleaved reps vs a83ac15, wu: 2048/16 1.062x, 4096/32 1.128x,
  8192/64 1.115x, 16384/8 1.106x.  Against 0d2ccdf (2 CTAs, lean) that adds about 2-4%: the third CTA is worth much
  less than the residency probe's 1 -> 2 step suggested.  The total moved within noise (+-0.5%) on the shared nmz5.

## 41. Every kernel, gfx938 against gfx936 (a83ac15 on both; bw52 vs nmz5; median of 3)
| kernel | 2048/16 gfx936 -> gfx938 | 8192/64 gfx936 -> gfx938 | share at 8192/64 on gfx938 |
|---|---:|---:|---:|
| s4b | 233.7 -> 247.2 us (0.95x) | 3119.6 -> 3346.3 (0.93x) | 23.8% |
| wu | 145.2 -> 138.7 (1.05x) | 1634.9 -> 1646.5 (0.99x) | 11.7% |
| dhu | 125.3 -> 80.9 (1.55x) | 1484.9 -> 1531.1 (0.97x) | 10.9% |
| s4a | 102.4 -> 109.8 (0.93x) | 1214.1 -> 1391.6 (**0.87x**) | 9.9% |
| prep_a | 115.3 -> 113.2 (1.02x) | 1251.7 -> 1390.4 (0.90x) | 9.9% |
| bwd_tail | 116.1 -> 96.0 (1.21x) | 1301.1 -> 1344.6 (0.97x) | 9.6% |
| fwd_h | 105.5 -> 79.1 (1.33x) | 1690.3 -> 1242.7 (**1.36x**) | 8.8% |
| prep_b | 75.4 -> 69.4 (1.09x) | 849.6 -> 918.6 (0.92x) | 6.5% |
| dav | 57.2 -> 55.0 (1.04x) | 686.8 -> 720.7 (0.95x) | 5.1% |
| gate | 34.0 -> 37.4 (0.91x) | 440.7 -> 480.0 (0.92x) | 3.4% |
| total | 1120 -> 1038 (1.08x) | 13734 -> 14060 (**0.98x**) | |
- At 8192/64 the same code is 2% slower on BW1102 than on BW1000.  Only fwd_h (bandwidth / latency recurrence)
  gains (1.36x).  s4a (0.87x), prep_a (0.90x), gate / prep_b (0.92x) and s4b (0.93x) sit between the compute ratio
  (0.78x) and parity: issue-bound kernels that the 1.5x HBM cannot help.  At 2048/16 the small-grid fixes of a83ac15
  (dhu 1.55x, bwd_tail 1.21x) make BW1102 8% faster overall.

## 42. s4a LDS restructure (P-938a, gfx938) + a fast A/B harness
- s4a's LDS (30464 B): S [128][68] (A staging, dvT, v tile, dwT), XT [64][68] (h double buffer in pass W, XC / AFS
  staging), KGs 2 x 2 KB (kgB tiles), BETAs.  P-938a: pass W writes dwT straight to HBM ([col][tr], 2 B per lane, the
  same values written once each) instead of staging it in S and bulk-copying at the end.  S is then free from pass W
  on and hosts the h double buffer, the kgB tiles and the dA-stage XC / A staging.  XT and KGs disappear: LDS 17664 B.
- Resources (gfx938): PF = 1 (production) VGPR 210 -> 2 CTAs per CU; PF = 0 VGPR 144 -> 3 CTAs per CU.
- Gate: 84/84 bit-identical with PF = 0 and with PF = 1.  s4a time (production PF = 1, 2 CTAs): 1.11-1.15x
  (2048/16 110 -> 98 us, 8192/64 1388 -> 1223 us, 16384/8 272 -> 238 us).  PF = 0 at 3 CTAs: 0.95-1.05x, so on gfx938
  the up-front operand prefetch is worth more than a third CTA (as with wu: the third CTA is worth little).
- gfx936 device code identical to a83ac15.  The KGs expression and the dwTg declaration keep their original form
  under !G2_S4A_LEAN; a first version that only re-spelled them changed gfx936 register allocation, which the
  per-kernel diff caught.
- Harness: kbench.py (one process per build, 2 warm-ups + 20 profiled fwd+bwd per shape, target-kernel median /
  min) with ab_fast.sh (two interleaved rounds).  A 3-shape A/B takes 2 minutes instead of about 60.  The earlier
  100-run sweep (one Python process per sample) cost about 100 minutes and was redundant: the direct A/B had
  already answered it.
- Noise check: unchanged kernels (dhu, fwd_h) differed by up to 17-34% between interleaved runs on the shared
  nmz5, so total-time ratios there are not evidence.  Decisions use per-kernel minima and require the rounds to
  agree.

## 43. P-938w2: wu's Wb / Qn straight to HBM instead of the LDS tile + wu_emit16 pass: bit-identical, 0.59-0.66x, reverted
- Positions computed as the exact inverse of wu_emit16 (wu_tile_dst), so the result is bit-identical (gate 84/84 twice).
- wu: 2048/16 130.3 -> 197.0 us (0.66x), 8192/64 1470 -> 2508 (0.59x), 16384/8 393 -> 656 (0.60x).
- Why s4a's direct dwT stores won (1.11-1.15x) and these lose: in s4a, 16 lanes write 16 consecutive bf16 (32 B
  runs per store instruction); here only 4 lanes form a contiguous 8 B run (the B-form scatter), and every lane
  issues 32 scalar 2 B stores.  The LDS tile exists exactly to turn that scatter into 16 B linear stores.
- Rule: replace LDS staging by direct stores only when a store instruction's lanes form >= 32 B contiguous runs.
  dav's dA (P-14: 16 B runs of 4 lanes) and dOT staging have the same scatter shape as here, so they are not tried.

## 44. s4b split (inter-chunk kernel X + intra-chunk kernel Y): cost model -> not worth it
- Bitwise feasibility: yes.  The fused epilogue (g2_wy.cuh, s4b_tail) combines per element, with contraction off:
  dqf = dqi + dq2; dk_out = dk + dk2b + dkt; dg_out = dg + qf dqf + (dk2b - dkt) kf, where dqi / dk / dg (incl. the
  cs1 / cs2 chunk-end terms and acc = A^T(-dw)) are inter-chunk / p3 terms and dq2 / dk2 / dkt come from the intra
  tables.  X can write dqi, dk, dg in fp32, and Y can evaluate the same expressions in the same order.  db also
  splits cleanly (xsum16(db_X) + xsum16(dbacc_Y)).
- Extra traffic per CTA: X writes 3 fp32 64x128 tiles (96 KB) and Y reads them back (96 KB); Y re-reads q, k (bf16)
  and g (fp32) for kf / qf / ep / en (about 64 KB).  Total about 256 KB per CTA, against s4b's about 255 KB per-lane
  reads plus 96 KB writes today, i.e. about 1.7-2x its per-lane bytes.  At 8192/64 (8192 CTAs): about 2.1 GB, about
  1.1 ms of HBM at about 1.9 TB/s, against s4b's 3.35 ms.  To break even, X + Y would have to be about 1.5x faster
  than the fused kernel on top of this.
- On gfx938 the binding resource is TA (per-lane bytes, section 17), not HBM bytes.  The split roughly doubles s4b's
  TA work; reading the intermediates by MLS (0.29x TA per byte) only helps the loads, not the per-lane stores.
- What the split would buy: lower VGPR / LDS per kernel, i.e. residency.  Measured on gfx938, a third CTA is worth
  2-4% (wu, section 40) or less than the up-front prefetch it displaces (s4a PF = 0, section 42).
- FLA's split for reference: at 2048/16 on gfx938, wy_dqkg + bwd_intra = 339 + 548 = 887 us, against our fused
  s4a + s4b = 365 us.
- Decision: do not split s4b (the estimated gain is far below the 8% threshold; the likely result is a loss).

## 45. Whole-chain validation on gfx938 (nmz5, 24 shapes, kbench: 10 profiled fwd+bwd per shape, 3 interleaved rounds, min)
- ba078ed gate 84/84 bit-identical.  ba078ed vs b3d03be: geomean **1.043x** (1.027-1.066x); vs a83ac15: 1.020x
  (0.994-1.031x).  No shape slower than b3d03be; 1024/8 within noise of a83ac15.
- Gains are uniform across T and H (3.1-6.6% vs b3d03be): H = 8 shapes get the most (5-6.6%: fwd_h 16-col slabs +
  dhu small grid), H = 64 the least (3.1-3.7%).
- Data: results/kda-gfx938-cards-20261025/fullval_b3d_a83_ba0.jsonl.

## 46. Not at the limit: re-plan around gfx938 (2026-10-26, user decision)
- Floors at 2048/16 on gfx938: theoretical 61 us, decomposition 388 us, fully loaded (current dataflow) 752 us,
  measured ba078ed about 1011 us.  The 4.3% so far approaches the *current dataflow's* floor; the 2.6x gap to the
  decomposition floor is a dataflow problem (6.3x the required bytes through TA, 13-15% of cycles on math).
- MLS in production today: only s4b's dA / dAqk staging (32 static instructions).  Not yet a gfx938 dataflow.
- H20 target (user): far above H20 on every size.  K3 data (2026-10-12): H20 FLA 16K / H48 = 18.36 ms; extrapolated
  16K / H64 about 24.5 ms against gfx938 ba078ed 26.4 ms (inference) -> about parity today, about 1.6x more needed for
  1.5x over H20.  H20 is unreachable right now (connection closed by the remote host); measure the 24 mk3 shapes
  (fwd / bwd split) when it is back.
- Principles: spend the spare bandwidth on the scarce compute and TA; every input to LDS by MLS with multi-stage
  prefetch (MLS uses no VGPRs, so the freed registers buy prefetch depth); LDS -> MMAC by ds_read_m /
  ds_read_matrix, no VALU reshuffles; inter-kernel tensors in MLS-readable blocked layouts.

## 47. s4a MLS prototype: design
s4a inputs per CTA (PF = 1 holds all of them in VGPRs up front: 88 KB / 256 threads = 88 VGPRs of the 210):
| input | bytes | layout today | MLS form |
|---|---:|---|---|
| A block | 8 KB | row-major, pitch RA | 4 tiles of 32x32 b16 (row stride RA) -> As tile-blocked; the aa extraction reads tile-blocked |
| dv (aN native) | 16 KB | 4 KB contiguous per wave | 2 tiles per wave; fragments by ds_read_b64 (dva stays in VGPRs: it is reused by 3 MMAC passes) |
| v rows | 16 KB | row-major, pitch R | 8 tiles (row stride R) -> the [64][136] v tile becomes tile-blocked |
| h blocks (pass W) | 32 KB | 8 x 4 KB contiguous | 2 tiles per kt, ring of 4 stages (16 KB) in S (free from pass W on: P-938a) |
| kgB tiles (pass W) | 16 KB | 4 KB contiguous per wave (8 kt x 512 B) | 2 tiles per wave once (or per-kt chunks of 8 rows) |
- Step 1 (largest, simplest): h blocks by MLS into a 4-stage ring in S.  Stage 0 stays the register prefetch
  phb[0]; stages 1-3 are issued at pass-W entry, stage kt+4 after the end barrier of iteration kt (3 iterations of
  lead, unlike P-938s).  Wave w < 2 issues tile w of each stage, so each issuing wave has exactly 3 younger MLS when it
  waits for stage kt: s_waitcnt vmcnt(3) before the barrier that publishes stage kt (younger dwT stores can only make
  that over-wait, never under-wait).  KGB moves behind the ring (S grows 17408 -> 20480 B; still 2 CTAs, which are
  VGPR-bound).  phb[1..7] leave the registers (-28 VGPRs).
- Step 2: kgB, v rows, dv, A by MLS (front phase), freeing the rest of the PF registers.
- Gates: bit-identical (same values, same MMAC order); s4a time by ab_fast (2 rounds, min).

## 48. s4a step 1 (P-938h): pass-W h blocks by MLS -- bit-identical, slower in both forms, reverted (diff kept)
| variant | resources (gfx938, PF = 1) | s4a vs ba078ed |
|---|---|---|
| 4-stage ring in S, stages 1-3 issued at pass-W entry, stage kt+4 after barrier kt | VGPR 146, LDS 20736 -> 3 CTAs / CU | 0.914-0.952x |
| 3-slot ring outside S, stages 1-3 issued at kernel start (same lead as the register prefetch) | VGPR 147, LDS 29952 -> 2 CTAs / CU | 0.732-0.834x |
- Both are bit-identical (two gate runs each).  Replacing 28 VGPRs of register prefetch (phb) with MLS made s4a slower
  even at 3 CTAs per CU and even with the same lead.  So the cost sits in the MLS path itself as used here: 2 waves
  issue 2 KB tiles, explicit vmcnt waits (which also wait for older dwT / dv2 stores), and LDS writes by the
  load path competing with the pass-W LDS reads.  Not the lead.
- Before any further MLS conversion: measure MLS inside a barrier-paced loop on gfx938, i.e. latency per 2 KB tile
  vs per-lane 16 B loads at the same bytes, with 2 / 4 issuing waves and with stores in flight.  The isolated
  streaming bench (section 18: MLS 0.29x TA, 431 vs 833 us) does not predict in-kernel behaviour.  SQTT of the
  P-938h kernel will show whether the waits sit on vmcnt (MLS latency) or on the barrier.
- Patch: results/kda-gfx938-cards-20261025/patches/p938h_s4a_mls_ring.diff.

## 49. MLS inside a barrier-paced loop (examples/kda/gfx938/mls_inloop.hip, gfx938, 8192 CTAs, s4a pass-W shape)
8 steps x 4 KB block, barrier per step, 8 MMAC + 8 ld8 LDS reads per lane per step, optional 4 dwT-like 2 B stores
per lane per step:
| variant | with stores | without stores |
|---|---:|---:|
| A per-lane 16 B loads + register prefetch | 280 us | 222 us |
| B MLS 2 KB tiles from 2 waves, 4-stage ring, 3 steps of lead | 318 us | **179 us** |
| C MLS 1 KB tiles from 4 waves | 304 us | -- |
| E = B with an exact vmcnt (lead kept, no over-wait) | 318 us | -- |
- Without concurrent global stores MLS wins (1.24x over per-lane loads).  With stores in flight it loses (0.88x):
  the stores add 58 us to the per-lane version but 139 us to the MLS version.  The exact vmcnt does not help, so it
  is not the wait count.  MLS and the global stores contend in the memory pipeline.
- This explains P-938h (s4a): its pass W issues the dwT stores (direct since P-938a) in every step, exactly the
  losing combination.
- Rule for gfx938: use MLS only in phases without concurrent global stores; keep outputs in LDS / registers and
  store them in a burst after the MLS phase (or stage MLS in a store-free phase).
- Next test: variant G = B with the stores deferred to one burst after the loop (LDS-staged output, the pre-P-938a
  dwT path).  If G beats A, s4a becomes: MLS h ring + LDS-staged dwT + end burst.
- Variant G (B with the stores deferred to an LDS buffer + one 16 B burst after the loop): 402 us; H (A with the same
  deferral): 289 us.  Caveat: G / H use 32 KB LDS (ring + output buffer) -> 2 CTAs per CU, against 4 for A-F (16 KB).
  At 2 CTAs per CU the MLS version loses badly (402 vs 289 us).  MLS needs many issuing waves per CU in flight;
  deferring the stores did not rescue it here.
- SQTT of s4a at 2048/16 (ba078ed vs ba078ed + P-938h, cycles %): VMEM issue 56.7 -> 37.1, waits 17.9 -> 34.2
  (vmcnt 2.6 -> 11.3, barrier 6.5 -> 13.2); cycles per wave +24%.  MLS removes the per-lane load issue stalls, but its
  completion latency reappears as vmcnt waits in the 2 issuing waves and as barrier waits in the other 2.  This is the
  same mechanism as in the microbenchmark.
- Conclusion for the MLS plan: in a CTA-cooperative, barrier-paced loop at 2 CTAs per CU (s4a / s4b / wu are VGPR- or
  LDS-bound there), MLS latency is exposed and only a few waves issue it.  MLS pays (1.24x) only with no concurrent
  stores and high CTA occupancy.  So the MLS rewrite must come with (a) every wave issuing MLS (no idle waves at the
  barrier), (b) store-free MLS phases, and (c) >= 4 CTAs per CU, which means much smaller per-CTA LDS / VGPR, i.e.
  smaller CTAs or work per CTA.

## 50. gfx938 (ba078ed) vs H20-3e FLA 0.5.2, 24 mk3 shapes, fwd / bwd split (mk3_fb.py on both, median)
- H20: h20-7 (new address 8.130.155.11:22), container tanbo_kda_h20; GPU 0 had about 10.6 GB free (other users hold
  132 GB per GPU), so 32768/64 ran out of memory there (blank, not zero).
- gfx938 / H20 total: geomean **0.871x** over 23 shapes (0.75-1.07x).  Slower on 19 shapes; faster only at H = 8 with
  T >= 8K (1.03-1.07x).  The forward is the weakest part (0.66-0.88x except H = 8), the backward 0.78-0.94x.
- Target (user: far above H20, taken as 1.5x): gfx938 needs another 1.41-2.00x (1.8-2.0x at H >= 32 and T <= 2K, about
  1.4x at H = 8, T >= 16K).  Data: results/kda-gfx938-vs-h20-20261026/.

## 51. Next phase (user go, 2026-10-26): forward first, small-CTA MLS dataflow
- Column H in the H20 table = KDA heads per GPU for 64 total heads (H = 64 / TP: 64 / 32 / 16 / 8 = TP1 / 2 / 4 / 8).
- Order: forward kernels first (gfx938 / H20 forward 0.66-0.88x): prep_a (10%, compute 34%), fwd_h (8.8%,
  recurrence), prep_b (6.5%, VMEM issue 45-50%, already 4 CTAs per CU at 16 KB LDS / 101 VGPR).
- Conditions under which MLS measured a win (section 49): every wave issues MLS (no idle waves at barriers), no global
  stores during the MLS phase, >= 4 CTAs per CU.
- Step 1: prep_b, the forward kernel that already meets the occupancy condition.  Inventory its input streams;
  restructure so all 4 waves issue MLS for their own tiles of the inputs, with stores grouped after the MLS / compute
  phase.  Gate bit-identical; ab_fast vs ba078ed.
- Step 2: prep_a (VGPR 139-163, LDS 21.5 KB -> 3 CTAs): split work per CTA (fewer rows or columns per CTA) to reach
  >= 4 CTAs, then the MLS input path.  Step 3: fwd_h (recurrence: MLS ring for the per-step K-side tiles, all waves
  issuing).

## 52. prep_b step 1 (P-938b / b2): R by wave-private MLS -- accepted (local commit on ba078ed)
- R4 = 4 strided 4 B loads per lane per 1 KB block; a wave needs 2 + 2I blocks (20 KB per CTA, more than the
  16 KB of smem), so each wave gets a private 4 KB slot (smem is free until the first barrier) and loads its blocks
  in 3 rounds of 2 KB MLS tiles (diagonal I..I+1 / 4+I..5+I, Aqk run, Akk run; tiles aligned to even blocks, all
  inside the CTA's 20 blocks).  `s_waitcnt lgkmcnt(0)` before a round overwrites the slot.
- Note: `matrix_load_32x16_b16` with stride 16 is not a 1 KB linear copy (gate failed on every tensor); only the
  32x32 form with stride 32 is the verified contiguous 2 KB copier.
- Result (gate 84/84 twice, prep_b target time, two interleaved rounds):
| shape | ba078ed | diagonal only (b) | all R (b2) |
|---|---:|---:|---:|
| 2048/16 | 68.6 us | 1.044x | **1.184x** |
| 8192/64 | 854 us | 1.045x | **1.076x** |
| 16384/8 | 216 us | 1.095x | **1.232x** |
- Even with 3 serial MLS rounds per wave it wins, as section 49 predicted: all 4 waves issue, no store in flight
  (the first store is Aqk after the loads), 4 CTAs per CU.  Whole-pipeline effect about +0.5-1.2% (prep_b 6.5%).

## 53. Forward vs H20 per kernel; prep_a sits at the VALU issue ceiling
Forward-only per-kernel device time (fwd_kern.py, us per call; gfx938 = f139032):
| 8192/64 | gfx938 | H20 FLA | ratio |
|---|---:|---:|---:|
| prep_a + prep_b (gate, intra A, solve, w/u) | 1381 + 968 = 2349 | gate 145 + intra_sub_chunk 459 + inter_solve 473 + recompute_w_u 361 = 1438 | **0.61x** |
| fwd_h (h recurrence + o) | 991 | h 473 + o 355 = 828 | 0.84x |
At 2048/16: prep 0.65x, fwd_h 1.07x.  At 16384/8: prep 0.70x, fwd_h 1.19x.  The forward gap is in prep.
- prep_a phase cut (G2_PREP_STOP, cumulative, 2048/16): loads + gate scan 44 us, gate exchange 52, QgA / Kb / Kn 79,
  P1a 91, P1b 113, all 116.  The sigmoid alone (expf + IEEE divide; dev build with it removed) = 16-17% of prep_a.
- MLS for the raw gate tile (16 KB into the free arena, P-938pa): bit-identical, 0.97-1.01x -> reverted.
  Packed-fp32 source for the differences / Kahan step (P-938pk): the compiler emits the identical ISA -> reverted.
- prep_a ISA (gfx938, GATE=1 VL=0 L2N=0): 3697 instructions per wave, fully unrolled (static ~ dynamic): v_sub 351,
  v_exp 241, v_mul 201, v_mov_b64 168, v_pk_mul 160, lshl 142, v_pk_add 137, fma 129, fmac 97, cvt_pk_bf16 96,
  mmac 80, div sequence ~10 per divide x 32, expf sequence ~13 x 32.
- Ceiling: BW1102 = 72 CUs at 1.3 GHz.  A 64-lane VALU op takes 4 cycles on a SIMD16.  8192/64 = 32768 waves over 288 SIMDs
  = 113.8 waves per SIMD x 3697 x 4 cycles = 1.29 ms vs measured 1.37 ms: **prep_a runs at about 94% of the
  instruction-issue limit.**  Only fewer instructions make it faster; occupancy / MLS / latency hiding cannot.
- Hardware ratio: H20 = 78 SMs x 128 fp32 lanes x ~1.98 GHz = 19.8 T lane-ops/s; BW1102 = 72 x 64 x 1.3 GHz = 6.0 T
  (12 T with packed fp32).  For VALU-bound work H20 has 1.65-3.3x the throughput, so a bit-identical prep cannot
  beat H20 by instruction trimming alone.
- Bit-preserving trims still available (each to be proven exhaustively on the input range, with a wave-uniform
  fallback): forced packed fp32 (inline asm) for differences / Kahan (~170 instr), the divide as rcp + 6 fma for
  d in [1, 2^126) (~100), expf without the overflow / underflow selects when the wave's arguments are in range
  (~128).  Total about 11% of prep_a, about 1% end to end.

## 54. Option A on prep_a (bit-preserving instruction trims) -- done, local commit cf1ad6c
| trim | status |
|---|---|
| packed fp32 for the differences / Kahan | **not possible**: gfx938 rejects neg / opsel on v_pk_add_f32 ("The opsel and neg field not support on this GPU"), and the compiler does not pack from source |
| P1a gm - G -> -(G - gm), the negation folded into v_exp (P-938ng) | done: 32 v_sub fewer per lane (32 v_exp with a neg source) |
| divide 1/d as rcp + 6 fma (numerator 1, no div_scale / div_fmas / div_fixup) | done: exhaustive on gfx938, identical for every fp32 d in [1, 2^126) (canaries detected); NaN payloads differ -> NaN goes to the reference path |
| expf without the overflow / underflow selects | done: exhaustive, identical for every fp32 \|x\| <= 87 and every NaN |
| `1 + e` as a VOP2 with inline constant (avoid v_mov_b64 {1,1} + s_nop) | dropped: gate mismatch, no time gain |
- Fast path selection: one wave-uniform test per CTA (max \|x\| <= 87 via v_max, NaN via a running sum), else the reference
  expression.  Production prep_a VGPR 139 -> 138, no spill.
- Result (gate 84/84 twice, prep_a target time, 2 runs of 2 interleaved rounds): 2048/16 0.99-1.00x, 8192/64
  1.04-1.07x, 16384/8 1.03x.  End to end about +0.3%.  prep_a stays at the VALU issue ceiling; A is exhausted here.

## 55. Why gfx936 did not take the algorithm-restructuring route (review)
- The rule set: R23 (A class, scheduling / data movement: bit-identical to production) and R23b (B class, user-confirmed
  2026-10-12: deterministic; fp64 error not worse than production; within FLA tolerance; K3 loss aligned).  On
  2026-10-17 the user tightened it: B1' (bf16 dq/dk handoff, +2.6-3.2%, dq/dk error 1.05-1.06x FLA) rejected.  Any
  algorithmic change must have **every tensor's fp64 error <= production's**, over several inputs and seeds.
- Under that gate, plan C was evaluated (results/kda-algo-explore-20261013/plan.md + probes).  F1 / F7 parallel scans and
  chunk 16 / 32 / 128 were excluded by traffic or work.  F3a was falsified (P-55).  F6 / C2 / F9 fusions were falsified
  by occupancy / contention.  Only F8 (16 x 16 sub-block solve) remained, at an estimated 2-4%.
- Reason it stopped there: on gfx936 the gap to H20 was bandwidth (H20 = 2.85x BW1000).  Restructuring the arithmetic
  does not remove bytes, so B could at most buy a few percent.  So B was "allowed under a strict gate, but no
  candidate paid".
- What changes on gfx938: HBM is no longer the limit.  prep_a is VALU issue bound (section 53), and H20 has 1.65-3.3x
  the VALU throughput.  Removing arithmetic now converts 1:1 into time.  The B candidates must therefore be re-scored
  by instructions removed, not by bytes.

## 56. prep_b phase cut, prep_a + prep_b fusion, gate / bwd_tail sigmoid -- and the 32 KB instruction cache
prep_b phase cut (G2_PREPB_STOP, cumulative, us):
| phase | 2048/16 | 8192/64 | 16384/8 |
|---|---:|---:|---:|
| R loads (MLS) | 13.3 | 98 | 31 |
| + Aqk / AqkA (8 x 8 B stores per lane) | 24.7 | 255-267 | 73-76 |
| + solve + Akk | 34.9 | 390 | 103-105 |
| + pass 0 (Kb -> Wn) | 45.9 | 513-590 | 137-141 |
| + pass 1 (v*beta -> Un), full | 60.5 | 790-935 | 177-187 |
Handoff ceilings (timing only): prep_b without the R / Kb loads 1.18-1.64x (-11 / -363 / -26 us), prep_a without
the Kb store + R copy 1.09-1.12x (-11 / -135 / -26 us): a fusion ceiling of 1.4-3.7% end to end.
Fusion P-938f (prep_a body + b_post in one CTA; R read from the arena, Kb from registers into T2): bit-identical
(gate 84/84 x2), but **0.75-0.90x** (VGPR 185 -> 2 CTAs).  R-only fusion (VGPR 157, 3 CTAs): **0.72-0.85x**.  Phase
cut of the fused kernel (8192/64): prep_a part 1277 us (as alone).  But the solve adds 541 us (130 in prep_b) and
pass 0 adds 720 us (160).  The b_post phases run 4x slower when they share the CU with prep_a code.
- Cause: **instruction cache**.  icache.hip on gfx938: a VALU stream costs 3.1 ns per instruction (4 cycles) up to 32 KB
  of code and 5.3-6.7 ns above it at 1 wave per SIMD.  The miss cost is only partly hidden at 2-8 waves (+8-15%) even
  with all waves in lockstep.  Code bytes (gfx938): fused 52.3 KB, prep_a 24.9 KB, prep_b 28.5 KB.  In the fused kernel
  the CU's waves execute two different 25-28 KB regions at once.
- Production gfx938 kernel code sizes: s4b 31.8 KB, bwd_tail 31.1 KB, prep_b 28.5 KB, prep_a 24.9 KB, wu 19.1 KB,
  gate 12.8 KB, s4a 11.8 KB, dav 9.2 KB, fwd_h 3.9 KB, dhu 3.0 KB.  s4b and bwd_tail sit at the 32 KB edge.  The
  bwd_tail sigmoid fast path (+~1-2 KB of code, same arithmetic) measured 0.88-0.97x: the same effect.
- Kept: the gate kernel's fast path (12.8 KB, gate 1.08-1.12x, commit 01540c8).  Fusion reverted (patch in
  results/kda-gfx938-fwd-20261026/patches/p938f_prep_fuse.diff).
- New rule (gfx938): code size is a resource like VGPR / LDS.  A kernel above about 32 KB, or a fused kernel whose
  concurrently executing regions exceed about 32 KB, pays instruction fetch.  Fusion needs the parts to shrink first
  (e.g. b_post's 4 per-wave template copies), and s4b / bwd_tail may gain from shrinking.

## 57. Correction to section 56: the instruction cache does NOT explain the fusion loss
- icache2.hip (gfx938, 2 waves per SIMD, even / odd waves of a CTA on two disjoint code regions):
| region per wave | working set | ns per instruction |
|---:|---:|---:|
| 8-16 KB, same region | 8-16 KB | 6.35 |
| 8-16 KB, split | 16-32 KB | 7.17-7.21 (+13%, the cost of two streams) |
| 20-24 KB, split | 40-48 KB | 7.66-7.92 (+21-25%) |
  Crossing 32 KB adds about 8-10% on top of the two-stream cost.  It cannot produce the 4x slower b_post phases.
- bwd_tail's fast-path slowdown was VGPR, not code: 114 -> 155 VGPR (4 -> 3 waves per SIMD).  Section 56's
  attribution to code size was wrong.  Section 56's "code size is a resource" rule is downgraded to "a second-order
  effect (<= ~10-25%) to check when kernels with disjoint per-wave code paths are fused".
- SQTT, 2048/16, R-only fusion vs the pair (cycles per traced wave, about 2.4 CTAs per prep_a / fused slot and 1.8 per
  prep_b slot):
| kernel | cycles / wave | ~cycles / CTA | sync % | wait_vm % |
|---|---:|---:|---:|---:|
| prep_a | 121398 | ~50k | 20.3 | 5.8 |
| prep_b | 63302 | ~35.6k | **42.7** | **21.4** |
| fused | 201194 | ~84k (= sum) | 22.4 | 8.1 |
  The fused CTA costs the sum of the two latencies.  Running prep_b's part inside a VALU-heavy CTA hides none of its
  latency, and its residency drops from 4 to 3 CTAs per CU.  Pair = N (a/3 + b/4); fused = N (a + b)/3.  The
  handoff bytes saved are smaller than the residency lost.
- Rule: fuse a latency-bound stage into another kernel only if the fused CTA keeps the latency-bound stage's
  residency.  Otherwise the saved handoff traffic is paid back in exposed latency.

## 58. bwd_tail (P-938c) and the gfx938-vs-H20 hardware ratio
- Production route (use_qk_l2norm_in_kernel=False) runs phase B as a plain dq / dk -> bf16 cast: 32 x (8 B load,
  4 B store) per lane, in a loop.  P-938c: 16 B loads (lane = 4 columns of one of 2 rows), all 16 of the warp's
  loads issued before the first convert, 8 B stores.  VGPR 114 -> 113, gate 84/84 x2, **bwd_tail 1.03-1.12x**
  (commit 71915eb).
- P-938c2 (issue those loads at kernel entry, under phase A): VGPR 173 -> 2 waves per SIMD, 0.92-0.94x -> dropped.
- Floor: bwd_tail moves 160 KB per CTA (reads dq, dk, dg 32 KB each + g_raw 16 KB; writes 3 x 16 KB).  8192/64 =
  1.31 GB.  Measured gfx938 streaming (hbm_bw.hip): read 1.80-1.87 TB/s, copy 1.49-1.61 TB/s.  So the floor is about
  0.85 ms vs 1.16 ms measured (73% of achievable).  At large shapes bwd_tail is now near its HBM roof.
- Measured bf16 MMAC peak (mmac_peak.hip, 16x16x16 lit_lts): 255 / 326 / 350 TFLOP/s at 1 / 2 / 4 CTAs per CU.
- gfx938 vs H20 by resource (H20 values: datasheet; not measured, H20 GPUs are full):
| resource | BW1102 (measured) | H20 | BW1102 / H20 |
|---|---:|---:|---:|
| HBM | ~1.85 TB/s read | 4.0 TB/s | **~0.46** |
| fp32 VALU | 6.0 T lane-op/s (12 TF fma, 24 packed) | ~44 TF | **~0.3-0.5** |
| bf16 matrix | ~350 TF | 148 TF | **~2.4** |
- Consequence for the plan: relative to gfx936, bandwidth was gfx938's surplus ("trade bandwidth for TA").  Relative
  to H20, bandwidth and VALU are both scarce and only the matrix unit is in surplus.  Beating H20 needs the work to
  move onto MMAC: recompute with MMAC instead of storing and reloading, and express elementwise or reduction steps
  as matrix products where the arithmetic stays exact.  Chain-wide: mmac 5% of cycles today (section 38).

## 59. s4b: read back or recompute -- per-CTA inventory and the first cut (P-938q)
Per CTA (one chunk x one head) before P-938q: reads 216 KB, writes 96 KB.  8192/64 = 2.56 GB in 3.33 ms
(48% of the ~1.6 TB/s mixed roof: s4b is latency / TA bound, so bytes matter where they sit on the wait path).
| tensor | KB / CTA | producer | alternative | verdict |
|---|---:|---:|---|---|
| A (Akk, bf16 64x64) | 8 | forward | -- | keep |
| dA (fp32 64x64) | 16 | s4a | needs s4a's chain | keep |
| dAqk (fp32 64x64) | 16 | dav | do . Vn^T on MMAC (1 MFLOP) but needs all 64 Vn rows in LDS (16 KB; S4BSmem is full); P-55 on gfx936 -18-25% | not now |
| do, v_new (bf16 64x128) | 16 + 16 | input / wu | v_new = U - W h needs U, W (32 KB): more bytes | keep |
| h, dh (bf16 128x128) | 32 + 32 | saved h / dhu | recurrence state: not local | keep |
| dwT (bf16 128x64) | 16 | s4a | -- | keep |
| g (fp32 64x128) | 32 | gate kernel | from g_raw = sigmoid + Kahan scan on the VALU (the scarce unit) | keep |
| q, k (bf16) | 16 + 16 | input | -- | keep |
| **out dq, dk (fp32)** | 64 (+64 read by bwd_tail) | -- | bwd_tail only casts them (no l2norm): write the final bf16 here | **done, P-938q** |
| out dg (fp32) | 32 (+32 + 16 g_raw read by bwd_tail) | -- | the in-chunk reverse cumsum + gate backward into s4b (VGPR 241, serial 16-row scan across lanes) | candidate, risky |
- P-938q (commit 9da4e3e): s4b stores bf16(dq), bf16(dk) (same bf_rne) into the final gradients; bwd_tail skips phase
  B.  Per CTA: s4b writes 96 -> 64 KB, bwd_tail reads 112 -> 48 KB and writes 48 -> 16 KB.  Gate 84/84 x2.
  **s4b 1.03-1.07x, bwd_tail 1.18-1.46x, fwd+bwd 1.016-1.050x** (2048/16 1.036, 8192/64 1.050, 16384/8 1.016).
  K3 (in-kernel l2norm) keeps the fp32 hand-off.  KDA_G2_S4B_BF16=0 (dev) restores it.
- Rule: a consumer that only re-formats (cast, copy, reshuffle) a producer's output is an interface, not a stage.
  Fold the re-format into the producer's store.

## 60. dg into s4b (estimate, not built) and the remaining re-format interfaces
dg into s4b -- what the tail does per column and what it would cost inside s4b (8192/64):
- The tail's phase A per column: a 16-row Kahan suffix scan (rows 15..0) per row block, the block carry (blocks above,
  fixed order), then per element: sigmoid, bdg, a Kahan dA accumulation and a Kahan ddt accumulation over the 16
  rows.  That is **three sequential 16-step chains per column**.
- In s4b a lane holds one row x 4 columns per kt, so all three chains run across the 16 lanes of a row block.
  - Via ds_bpermute: 16 x ~120 cycles x 8 kt x 3 chains, about 46k cycles per CTA vs ~19k cycles per CTA today: >2x.
  - Via DPP (row_shr within 16 lanes, ~8-12 cycles): about 3-5k cycles (+15-25%).
  - Plus ~800 VALU per lane for the gate math (+~16% of s4b's VALU).
  - Plus g_raw loads, with VGPR at 241 of 256 (any spill is fatal).
- The alternative (transpose the dg tile through LDS to the tail's lane-per-column mapping) needs 4-16 KB of LDS.
  S4BSmem is full.
- Savings: bwd_tail now 787 us (after P-938q) would shrink to the 13 us reduce.  s4b writes dg as bf16 dg_raw (-16 KB
  / CTA) but reads g_raw (+16 KB).  Net (rough): -787 us (tail) + 0.15-0.25 x 3190 us (s4b) = -0 to -310 us, i.e.
  0-2.3% end to end.  The VGPR edge makes it likely negative.  **Not pursued.**
Remaining "re-format only" interfaces (rule of section 59):
| producer -> consumer | tensor | what the consumer could read instead | best-case saving (8192/64) |
|---|---|---|---|
| dav -> dhu | dOT (bf16, 16 KB / CTA) = do re-laid as dhu's B operand | do itself (row-major), via MLS + ds_read_m transposed B | dav store 134 MB, about 84 us (0.6%), minus the dhu cost of reading the strided layout |
| prep_b -> fwd_h | AqkA (bf16, 8 KB / CTA) = Aqk (saved for the backward) in fwd_h's A-fragment order | Aqk row-major: same 8 B per lane, 32 B segments per row instead of 512 B per wave | 67 MB, about 40 us (0.3%) |
- The backward gate kernel (445 us, recomputes g from g_raw) would disappear if the forward saved fp32 g.  But that
  costs +512 B per token-head of saved memory, beyond the approved budget (FLA default 1032 B/token-head).  Out.

## 61. Step 1 of the next round: where HEAD (8bd1ff8) stands, and the per-kernel headroom
Pushed to the fork: ba078ed..8bd1ff8 (f139032, cf1ad6c, 01540c8, 71915eb, 9da4e3e + 8bd1ff8).  8bd1ff8 keeps the
G2_PREPB_STOP dev returns out of the gfx936 build.  All 129 gfx936 kernel bodies are identical to ba078ed (only the
kernarg sizes of prep_b / s4b / bwd_tail grew with the appended fields).
24 shapes, fwd / bwd split (mk3_fb.py, 2 processes per shape, median; H20 = the 2026-10-26 FLA run):
- HEAD vs ba078ed: **geomean 1.059x**, every shape faster (1.037-1.127x).
- gfx938 / H20: **0.871x -> 0.924x** (23 shapes; 32768/64 has no H20 point).  Above H20 now: 4096/8 (1.01),
  8192/8 (1.07), 16384/8 (1.10), 32768/8 (1.12).  Lowest: H = 64 (0.83-0.87) and T = 1K (0.83-0.86).  The forward
  stays the weaker half (0.76-0.79 at H = 64).
- Needed for 1.5x H20: 1.34x (32768/8) to 1.81x (1024/64) more.
Ceiling cards re-collected on HEAD (results/kda-gfx938-cards-20261027-hd1, cards_cmp.py), headroom = t - floor (the
card floor = the current dataflow's busiest resource at its measured peak):
| 8192/64 | t (us) | floor | t / floor | headroom | primary | TA / VALU / MMAC busy |
|---|---:|---:|---:|---:|---|---|
| s4b | 3188 | 2046 | 1.56 | **1142** | TA | 64 / 50 / 18% |
| wu | 1469 | 1137 | 1.29 | 332 | TA | 76 / 61 / 6% |
| dhu | 1513 | 1186 | 1.28 | 327 | TA | 76 / 29 / 9% |
| fwd_h | 1225 | 1008 | 1.22 | 217 | TA | 79 / 32 / 12% |
| prep_a | 1312 | 1096 | 1.20 | 216 | VALU | 48 / 81 / 17% |
| dav | 767 | 593 | 1.29 | 174 | TA | 76 / 46 / 6% |
| gate, prep_b, bwd_tail, s4a | 444, 887, 779, 1219 | -- | 1.03-1.15 | 40-59 each | -- | at their floors |
At 2048/16 and 16384/8 the order is the same: s4b first (87 / 317 us of headroom, 1.62 / 1.65x), then wu
(50 / 99 us).  s4b holds about 40% of all remaining in-dataflow headroom.  Its busiest unit is only 58-64%
busy, so it is latency / serialisation bound: the waits and barriers of the kt loop.

## 62. s4b kt loop: where it waits (SQTT by instruction), and why the next-kt prefetch loses
SQTT of s4b on HEAD (8192/64, 32 traced waves ~ 3.4 CTAs each; stall_ctx.py: stall charged to the waiting
instruction, keyed by its text + the 2 preceding instructions; the SQTT "offset" field is not stable across waves).
- Cycles by class: VALU 25.4%, **vmcnt waits 15.3%**, VMEM-load issue 14.2%, LDS 10.1%, MMAC 9.3%, lgkm waits 9.1%,
  SALU 7.4%, **barriers 6.8%**, stores 1.2%, MLS 1.0%.
- The largest single waits are the four per-wave-variant `s_waitcnt vmcnt(3)` right after the first MMAC block of
  every kt, ~6.2% together.  The loop issues dwT, h/dh(kt+2), g, gn, k, q in that order; vmcnt(3) waits for g (its
  shfl for g8), then vmcnt(2) for gn (the exp2 of gn - g) and vmcnt(1) for k.  So the per-kt row operands are
  loaded and used within the same kt.
- LDS budget: S4BSmem is exactly 32768 B (= the 2-CTA limit), and VGPR is 241 of 256.  The table double buffer
  (+13 KB) and an LDS copy of the gn row (+0.5 KB) do not fit.
Prefetch of the next kt's row operands (bit-identical in every form, gate 84/84 x2):
| variant | VGPR (dense / varlen) | s4b 2048/16, 8192/64, 16384/8 |
|---|---|---|
| P-938k g, gn, k, q for kt+1, issued after this kt's dwT / h / dh loads | 253 / 256 + 40 B spill | 0.969, **0.883**, 0.998 |
| g, gn only, same position | 249 / 256 + 40 B spill | 0.964, 0.914, 0.953 |
| g, gn, k, q issued after this kt's stores (end of iteration) | 253 / 256 + 36 B spill | 0.963, 0.886, 0.972 |
- SQTT of the second variant: vmcnt waits rise 15.3% -> 26.0%; the new `s_waitcnt vmcnt(1)` sit right behind the
  prefetch loads.  **vmcnt retires in order**: a prefetch issued before this kt's own loads must land before any of
  them can be consumed, so it serialises the kt instead of overlapping it.  Moving the prefetch to the end of the
  iteration removes that, but stays 0.89-0.97x.  The extra 12 live VGPRs (253 of 256) plus the varlen spill cost
  more than the hidden latency is worth at 2 waves per SIMD.  Reverted.
- Rule: on gfx9-class vmcnt, a software prefetch must be the youngest outstanding load at every wait that precedes
  its use.  Place it after the last load consumed before it, or keep it in a buffer the hardware fills (LDS via
  MLS) that does not join the vmcnt order of the per-lane loads.
- s4b's remaining levers are structural: fewer per-kt loads (the gn row once per CTA, dwT for all kt), and the
  second barrier per kt.  Both need LDS that does not exist at 2 CTAs per CU, i.e. a smaller S4BSmem (e.g. h / dh
  staging single-buffered, or the tables in bf16 -- not exact).

## 63. wu and dhu by the same method -- and the gfx936-tuned host rules re-measured on gfx938
SQTT by instruction (stall_ctx.py, HEAD 8bd1ff8):
| kernel | top classes | the largest waits |
|---|---|---|
| wu 8192/64 | VALU 30.9%, **vmcnt 24.0%**, lgkm 11.6%, SALU 7.2%, LDS 6.9%, barrier 6.6% | `s_waitcnt vmcnt(0) lgkmcnt(0)` + `s_barrier` (__syncthreads) ~10% in total |
| dhu 8192/64 | **stores 20.1%**, **barrier 19.9%**, VALU 15.1%, loads 14.1%, LDS 10.8% | `global_store_dwordx2` 11.6% and `global_store_short` 6.8% issue stalls; barriers 19.7% |
| dhu 16384/8 | VALU 20.8%, barrier 16.1%, LDS 15.4%, loads 14.6% | barriers 15.8% (32 CTAs, chain bound) |
- wu P-938y: every `__syncthreads()` -> `lds_barrier()` on gfx938.  Safe: wu re-reads no global value it wrote and
  fills no LDS by matrix_load.  1.035 / 1.000 / 0.976x (2048/16, 8192/64, 16384/8): the vmcnt(0) were mostly real
  data waits that just moved.  Reverted.
- dhu: production is `g2_dhu_v8b_kernel<1, false>`.  The aN(dv2) emit is 4 scattered 2 B stores per lane, because
  the linear 8 B emit (P-22d) was gated to <= 96 CTAs from gfx936 measurements.  **P-938l (commit 6edce16): linear
  at every grid on gfx938: dhu 1.17x at 4096/32, 1.30-1.37x at 8192/64, neutral at 2048/16 and 16384/8** (both
  measurement orders; gate 84/84 x2).
Sweep of the other host thresholds on gfx938 (dev knobs, kernel time):
| knob | gfx936-tuned default | gfx938 result |
|---|---|---|
| **KDA_G2_S4A_AFS_MAXH** (staged-A s4a) | 12 | **AFS at every H: s4a 1.22-1.35x at H = 16 / 32 / 64** -> P-938s4 (commit f814bb6) |
| KDA_G2_DHU_THROTTLE_MIN_CTAS | 192 | off: 0.93x at 4096/32, 1.04x at 8192/64 -> keep |
| KDA_G2_V8O4_THROTTLE_MIN_CTAS (fwd_h) | 224 | off / 64: within noise -> keep |
| KDA_G2_SGPR_MAX_CTAS (fwd_h) | 80 | 0: 0.95x at 2048/16; all: 0.85x -> keep |
| KDA_G2_V8O4W_MIN_H / _NT (wide fwd_h) | 40 / 512 | everywhere: 0.78-0.84x at 2048/16 and 4096/32 -> keep |
| KDA_G2_R1A (fwd_h 4-wave dedup) | 1 | 0: 0.56-0.86x -> keep |
| KDA_G2_WU_ORDER | 2 (H % 32 == 0) / 3 | 0: 0.69-0.99x, 3: 0.89-1.0x -> keep |
| KDA_G2_R1A_DHU (4-wave dhu) | auto (96-127 CTAs) | forced: 0.79-0.91x -> keep |
- Whole fwd+bwd, f814bb6 vs 8bd1ff8 (s4a + dhu rules, both measurement orders): 2048/16 1.016-1.017x, 4096/32
  1.029-1.066x, 8192/64 1.036-1.076x, 16384/8 neutral (H = 8 already had both forms).
- Measurement note: ab_fast has a position bias (the same build measured 3-5% apart at 16384/8 in positions 1 vs 2 / 4).
  Decisions near 5% now use both orders.
- Lesson: every gfx936-measured host threshold is a candidate on a new card.  Two of nine flipped, and they were worth
  more than most kernel rewrites of this round.

## 64. s4b structural rewrite -- evaluated by ceilings first (not built)
S4BSmem = TA / TQ / TK 13056 B + gM 256 + cs1 / cs2 1024 + stg (h, dh x 2 buffers) 16384 + dwTs 2048 = 32768 B.
- h / dh single buffer: today tile kt+2 is loaded at the top of kt and written to stg[buf] after the first barrier.
  A single buffer would load tile kt+1 at the top of kt and write it at the same point.  No h / dh read follows the
  first barrier of a kt, and the second barrier orders the write before kt+1's reads.  So the load -> LDS distance
  is unchanged: **zero extra latency, exact, 8 KB of LDS freed inside the loop**.  (The prologue's dA / dAqk MLS
  staging still uses all 32 KB, so the allocation stays.)
- What the 8 KB could buy, bounded by timing-only ceilings (gfx938, quiet GPU 5, f814bb6):
| ceiling (numerically wrong, timing only) | s4b 2048/16, 8192/64, 16384/8 | what it bounds |
|---|---|---|
| c1: no per-kt g / gn loads | 1.000, 1.040, 1.028 | the g tile (+ gn row) by MLS into a double buffer (exactly 8 KB) |
| c2: no per-kt g / gn / k / q loads | 1.061, 1.099, 1.075 | all row operands via LDS (needs ~16 KB: does not fit) |
| c3: no second barrier per kt | 1.000, 0.975, 1.000 | table double buffer (+13 KB: does not fit) -- the lever is worth nothing |
- Verdict: the barrier is not on the critical path, and the row-operand loads are worth at most 6-10% of s4b
  (<= 2.4% end to end at 8192/64).  The part that fits the freed 8 KB is worth at most 0-4% of s4b, before the MLS
  costs (stores in flight every kt: section 49).  **The structural rewrite does not pay**; s4b's remaining
  headroom vs its card floor is VALU / LDS / TA issue spread across the whole kt body, not one wait.

## 65. 24 shapes on f814bb6 (pushed: 8bd1ff8..f814bb6)
- First run invalid: nmz5 was shared during it (GPU 3 with another process holding 24% VRAM, GPUs 6 / 7 training a
  VLA model, host load 45-123).  8 shapes came out 0.56-0.94x, which contradicts every per-kernel A/B.
- Rerun on the idle GPU 4: f814bb6 vs 8bd1ff8 1.02-1.08x at H >= 16 (H = 8 neutral, as expected: both rules were
  already active there).  The T = 32K tail of that run was still disturbed, so it was re-measured interleaved
  (hd1 / af1 alternating, 2 samples each): 32768/64 1.049x, /32 1.055x, /16 1.025x, /8 0.994x.
- **gfx938 / H20 (23 shapes): 0.871 (ba078ed) -> 0.924 (8bd1ff8) -> 0.947 (f814bb6).**  Above H20: 8192/8 1.07, 16384/8 1.11,
  32768/8 1.11, 32768/16 1.01; close: 4096/8 0.99, 16384/16 0.99, 32768/32 0.98, 16384/32 0.98, 8192/32 0.97.  Backward is
  now at 0.94-1.01 for H >= 32 and T >= 8K.  The forward stays the gap: 0.76-0.78 at H = 64.
- Rule for the measurements: check hy-smi / load before and after every chain run; rerun on an idle card any run
  whose shapes disagree with the per-kernel A/B.

## 66. dav (P-938d, commit 9133470, pushed) and fwd_h (ceilings; nothing kept)
SQTT by instruction (8192/64): dav = vmcnt waits 32%, VALU 18%, SALU 16%, loads 13%, barrier 6.7%; the big waits
are the `vmcnt(0)` at the start of each pass (operands loaded at the top of their own pass).  fwd_h = load issue
28.9%, VALU 17%, LDS 12% + lgkm 12%, stores 9.7%, barrier 6.7%.
- **dav P-938d**: all global operands of both passes issued before pass 1 (pass 1's do rows, pass 2's do tile, the
  A^T gathers); the two loops that index them unrolled.  VGPR 74 -> 106, still 4 CTAs per CU, no scratch, gate
  84/84 x2.  dav ~1.06x / 1.10x / 1.13x (2048/16, 8192/64, 16384/8; mean of both orders -- the second position
  measured 3-8% slow in this pair).
- fwd_h ceilings (timing only, both orders):
| ceiling | fwd_h 2048/16, 4096/32, 8192/64, 16384/8 |
|---|---|
| cA: CTAs vs != 0 skip the K-side loads shared by a head's 4 slabs (W, QgA, AqkA, G, K) | 0.86, --, 0.66-0.91, 1.00 (slower) |
| cB: no o and no HbN stores | 1.00, --, **1.38**, 1.00 |
| cB1: no HbN stores | --, 1.03-1.17, 1.02-1.18, -- |
| cB2: no o stores | --, 1.13, 1.16, -- |
- P-938o (o staged per wave as 16 rows x 32 columns -> one 16 B store per lane, 64 B per row, instead of two 8 B
  stores, 32 B per row): bit-identical, 0.95-1.02x -> reverted.  The store cost at 256 CTAs is the write traffic
  of partial lines (each CTA owns 32 of a row's 128 columns), not the instruction count.  The HbN stores are
  already 512 B contiguous per instruction (saved h, inside the approved memory budget).  fwd_h stays as is.

## 67. 24 shapes on 9133470 (GPU 4; GPUs 6 / 7 were training other jobs during part of the run)
- 9133470 vs ba078ed: geomean **1.078x** (24 shapes, 1.04-1.13x); vs 8bd1ff8 1.018x.
- gfx938 / H20: **0.937** (23 shapes; ba078ed 0.871).  At / above H20: 8192/8 1.08, 16384/8 1.11, 32768/8 1.11,
  32768/16 1.00; 32768/32 0.99, 4096/8 0.99.  Backward 0.95-1.02 at T >= 8K; forward 0.70-0.83 at H >= 32.
- Run-to-run noise on the shared nmz5 is about +-4% per shape (f814bb6's run gave 0.947 with 32K re-measured;
  several shapes moved 3-6% between the two runs with no relevant code change).  Cross-time chain numbers are
  therefore only good to a few percent; release numbers should come from one interleaved run (ba078ed / HEAD /
  same-card FLA alternating per shape) on a quiet node.

## 68. Forward refocus: prep_b SQTT; prep_a B class trial B1 (fast gate sigmoid)
- Forward per kernel (9133470) vs H20 FLA (h20-7 data): prep_a + prep_b 0.64 / 0.69 / 0.71x of FLA's gate + intra +
  inter_solve + w/u (2048/16, 8192/64, 16384/8); fwd_h 1.07 / 0.85 / 1.18x.  The forward gap is prep.
- h20-5 (new node, 8 idle H20-3e): container tanbo_kda_h20 on nvcr pytorch:26.09 (torch 2.14 / Triton 3.8).  FLA 0.5.2
  (fla-core + flash-linear-attention) runs 7-9% slower forward / ~3% slower backward there than on h20-7's torch
  2.10 / Triton 3.6 stack; the identical stack is being installed into /home/locauser/kda_site (slow mirror).
- prep_b SQTT (8192/64): vmcnt waits 27.8% (the three serial R MLS rounds ~9%, pass 0 / 1 operand waits), barriers
  16.2% (wave imbalance of the per-wave triangular solve: wave I does 2 + 2I blocks), stores 14.0%, MLS 7.1%.
- **B1 (B class): the safe-gate sigmoid as bare v_exp_f32 + v_rcp_f32** in prep_a, the backward gate kernel and
  bwd_tail (consistently), gfx938 only.  Speed (both orders): prep_a 1.07-1.12x, gate 1.16-1.54x, bwd_tail
  1.32-1.39x, **fwd+bwd 1.031-1.047x**.
- Precision gate (ft_fp64_gate.py: frozen FP64 oracle, K3 route, 13 cases = 10 seeds at 4096/12 + 8192/48 +
  16384/12 + 32768/4; production 9133470 vs B1 on identical inputs): deterministic (two runs bit-equal) for both.
  Per-tensor geomean error ratio B1 / production: o 1.0000, q 1.0001, k 1.0001, v 1.0001 (t <= 1.9, |delta| <=
  0.07%), g_raw 0.9986, beta_raw 1.0005, A_log 0.962, dt_bias 0.988 (the last two are the noisy reduced grads,
  R25).  So rules 1-2 hold (no tensor significantly worse).  Rule 3 (FLA tolerance) is unchanged in practice.
  Rule 4 (Megatron K3 loss curve) not run yet.  Patch: results/kda-gfx938-bclass-20261028/p938b1_fast_sigmoid.diff
  (not committed: needs the user's B-class decision; gfx938 outputs would then differ from gfx936's in the last bits).

## 69. B1 committed (f7f2170); Megatron K3 loss check on nmz5; P-938b3 dropped
- FLA's own gate sigmoid on NVIDIA is the same class as B1: `tl.sigmoid = 1 / (1 + tl.exp(-x))` compiles to
  `ex2.approx.f32` + `div.full.f32` (PTX checked on h20-5).  gfx936 production never used a fast sigmoid: P-26 / P-26b
  were measured there (VALU -54% in the gate kernel, -20% in prep_a, but only 1.004-1.016x on the bandwidth-bound
  card; precision neutral over 10 seeds) and rejected by the bit-exact rule (R23).
- Megatron-LM-k3, Yu Hui's 3-layer recipe on nmz5 (8 x BW1102, TP2 / EP4 / DP2, seq 2048, Muon, mock data, 30 iters;
  NIC ens19f0, no NUMA binding, emerging-optimizers 0.3.0; Yu Hui's blas nightly path is not visible on nmz5),
  lm loss at iteration 30 (two sessions):
| arm | runs | loss @30 | median ms / iter |
|---|---|---|---|
| FLA (Triton on gfx938) | 2 | 8.023, 7.991 | 1956, 1999 |
| flash-train 9133470 (exact sigmoid) | 4 | 8.042, 8.146, 8.172, 7.914 | 1881-1960 |
| flash-train f7f2170 (B1) | 4 | 7.499, 7.448, 7.451, 7.914 | 1891-1953 |
  The same build repeated spans up to 0.26 (production) / 0.47 (B1) at iteration 30: the run is not deterministic
  and the trajectory amplifies tiny differences (iteration 1 losses: FLA 10.8036, production 10.8007, B1 10.8033).
  B1 sits inside or below the production / FLA band (never above), and the iteration times are equal within noise.
  Rule 4 (no training regression) is met to the resolution of this setup.  The absolute loss level differs from
  bw61's gfx936 run (6.71 at iteration 30), so this check is relative only.
- P-938b3 (prep_b: the Aqk off-diagonal run by per-lane gathers in flight with the diagonal MLS round, 2 MLS rounds
  instead of 3): bit-identical to the B1 golden, but prep_b 0.97-1.00x on a quiet node (both orders) -> reverted.

## 70. Clean gfx938-vs-H20 baseline on f7f2170 (2026-10-29; both sides one GPU, sequential)
- Protocol: mk3_fb.py, 24 shapes x 3 processes, median; nmz5 GPU 0 (build /root/fthead = f7f2170) and h20-5 GPU 0
  (FLA 0.5.2, torch 2.10 / Triton 3.6, kda_site).  H20 reps agree within 0.5% and match the earlier single runs
  (2048/16 fwd 0.204, 8192/64 fwd 2.445 ms); 32768/64 now has an H20 point (H20-3e, 143 GB).  The 10-28 run (8
  shapes in parallel per node) is void: H20 times doubled under the parallel load.
- nmz5's host load rose to ~250 (another user's compile jobs) during the last ~20 min; some single reps are outliers
  (16384/16 fwd 2.07 vs 1.57 ms), the medians are not affected.
- **gfx938 / H20 = 0.942 (24 shapes; fwd 0.829, bwd 0.982)**; at or above H20: 4096/8, 8192/8, 16384/8, 16384/16,
  32768/16, 32768/8.  The gap is the forward: 0.69-0.75 at H = 64.
- Per kernel (8192/64): forward 3222 vs 2452 us = prep_a 1267 + prep_b 960 + fwd_h 995 against gate 160 + 61,
  intra_sub_chunk 508, inter_solve 529, w/u 362, h 475, o 356.  Backward 9806 vs 9897 us.
- Note: bwd_kern.py does not clear .grad between its backward calls, so both sides list `vectorized_elementwise`
  (gradient accumulation, ~10% of their backward lists).  mk3_fb clears the grads; the totals do not contain it, and
  a leaf / non-leaf probe shows no non-KDA kernel in one flash-train backward.

## 71. Forward hand-off ceilings on gfx938 (the R63 check before any hand-off work)
Timing-only variants of f7f2170 (numerically wrong), ABBA with warm-up on one GPU each:
| ceiling | prep_b | prep_a | total |
|---|---|---|---|
| c1: R without the Aqk half (prep_a dumps 10 of 20 blocks; prep_b skips the Aqk MLS tile / run) | 1.03-1.08x | 1.005-1.017x | 1.002-1.005x |
| c2: no AqkA store (the duplicate A-native copy of Aqk) | 1.04-1.08x | -- | 1.002-1.004x |
| c3: c1 + c2 | 1.12-1.15x | 1.005-1.012x | **1.006-1.009x** |
- Unlike gfx936 (section 11 of kda_mk3_round.md: 0.1-0.7%), prep_b does respond, but the whole is below 1%, before
  the real costs (prep_a writing Aqk / AqkA; fwd_h or dav reading the other layout).  Not built.

## 72. prep_a instruction census (gfx938, static = dynamic: fully unrolled)
- f7f2170 prep_a: 3083 instructions per wave; VALU fp 870, int / address 596, mov 267, trans 273, cvt 156, MMAC 80.
  By phase (stop markers): loads + gate scan 3248 VALU cycles, gate exchange 1436, QgA / Kb / Kn 3308, P1a 2232,
  P1b 3040.
- **P-938sd**: with B1 on, the P-938sg range test (abs / max / sum per element + a ballot) was still computed in
  prep_a and the gate kernel although nothing read it.  **P-938ga**: prep_a's 16 raw-gate loads per lane use one
  chunk base + 32-bit unsigned lane offsets instead of per-lane 64-bit addresses (in the gate kernel the same change
  added instructions -> not applied there).  prep_a 3083 -> 2971, gate 816 -> 779.  Bit-identical.
- prep_a 1.03-1.04x at 2048/16, 4096/32, 16384/8, 32768/16, but 1.00x at 8192/64: at H = 64 prep_a is not VALU
  bound.  That led to section 73.
- The algebraic B-class trim of P1b (eBo = exp2(gr_rb - G_cb) factored per sub-block) saves ~1% of prep_a: every lane
  owns one row per sub-block, so per-element and per-column forms cost the same number of instructions.  Not built.

## 73. Per-head normalisation (R60) on gfx938: the CTA visiting order (P-938o2, commit a02c016)
Equal work (8192 CTAs, T x H = 524288), clean run of section 70:
| kernel | 8192/64 | 32768/16 | H = 64 extra | H20 FLA |
|---|---:|---:|---:|---|
| gate (backward) | 509 | 259 | +97% | 189 / 186 |
| dav | 768 | 566 | +36% | -- |
| prep_b | 960 | 732 | +31% | flat |
| bwd_tail | 627 | 525 | +19% | -- |
| wu | 1594 | 1453 | +10% | flat |
| prep_a | 1269 | 1189 | +7% | flat |
- First try, head-fastest for prep_a / prep_b (as P-88 did for s4a / s4b): **0.61-0.73x at H = 64**.  Their outputs
  (R, Wn, Un, QgA, Kn, AqkA, Gn) are head-major natives; head-fastest makes the co-resident CTAs write blocks NT x
  blk apart.  R made chunk-major as well: still 0.64x (the other natives remain).
- Cause, as for wu's P-95: with chunk-fastest order the co-resident CTAs (consecutive chunks of one head) advance the
  row-major streams by 64 rows x H x 256 B = 1 MiB per chunk at H = 64, a power of two, in lock-step.
- g2_cta_order (shared; wu's orders 0-3) on prep_a, prep_b, gate, bwd_tail, dav; sweep (3 rotated rounds, 12 shapes,
  kernel time, speedup over order 0):
| kernel | streams | head-fastest (1) | diagonal (2) | rule (gfx938) |
|---|---|---|---|---|
| gate | row-major in / out | **1.54-2.09x at H = 64**, 1.36-1.63x at H = 32, mixed at H <= 16 | 1.52-1.84x | 1 at H % 32 == 0 |
| bwd_tail | row-major | 1.03-1.16x at H = 64, 1.0-1.09x at H = 32 | 1.02-1.14x | 1 at H % 32 == 0 |
| prep_b | row-major in, head-major natives out | 0.61-0.65x | **1.08-1.33x at H = 64**, ~1.0 at H = 32 | 2 at H % 64 == 0 |
| dav | row-major do / A, p3 natives | 0.55-0.64x | 1.07-1.20x at H = 64, 0.96-0.98x at H = 32 | 2 at H % 64 == 0 |
| prep_a | row-major in, head-major natives out | 0.71-0.83x | 1.02-1.07x at H = 64 | 2 at H % 64 == 0 |
  Order 3 (M = 4 / 8 heads per chunk) was within +-5% of order 0 everywhere.  gfx936 keeps order 0 (not measured).
- With P-938sd / ga (build o4 vs f7f2170, ABBA, fwd+bwd per iteration): H = 64 1.034x (1K), 1.046x (2K), 1.056x (4K),
  1.064x (8K, 16K), 1.048x (32K); H = 32 1.006-1.022x; H <= 16 within +-2%.  Gate 84/84 bit-identical x2.
- Rule R64 (gfx938): pick each kernel's CTA order by its stream families.  Row-major only -> head-fastest; row-major
  in plus head-major natives -> diagonal; never head-fastest when the kernel writes head-major natives.  Check every
  kernel by per-head normalisation (equal T x H, different H) whenever H is a power of two >= 32.

## 74. 24 shapes on a02c016 (build o4), same protocol as section 70 (nmz5 GPU 0, host load 7-17)
| T | H = 64 | H = 32 | H = 16 | H = 8 |
|---|---|---|---|---|
| 1K | 0.854 | 0.858 | 0.815 | 0.839 |
| 2K | 0.905 | 0.910 | 0.873 | 0.908 |
| 4K | 0.873 | 0.957 | 0.937 | 0.993 |
| 8K | 0.957 | 0.973 | 0.987 | **1.047** |
| 16K | 0.972 | 0.990 | **1.012** | **1.093** |
| 32K | 0.985 | **1.028** | **1.031** | **1.118** |
(gfx938 / H20, fwd + bwd device time.)
- **gfx938 / H20: 0.942 (f7f2170) -> 0.952 (24 shapes); forward 0.846, backward 0.988.**  a02c016 vs f7f2170 across
  the two runs: H = 64 1.036-1.057 (1K: 1.007), H = 32 1.010-1.030, H = 16 1.00.  H = 8 at T <= 4K read 0.95-0.98
  here, but the interleaved ABBA of the same pair gave 0.995-0.998 there (H = 8 takes none of the new orders):
  cross-run noise.
- Backward is at or above H20 for T >= 4K at every H >= 16 (1.01-1.07); the forward stays 0.70-0.84 at H >= 32
  (prep_a VALU, prep_b TA, fwd_h writeback; sections 53-66, 71-72).
- Next candidates, by size: (1) the forward prep at H <= 32 is VALU / TA bound with A-class trims exhausted
  (needs B-class restructuring of prep_a's gate / weighting, or a different prep split); (2) bwd_tail's serial
  512-add CTA dA sum on wave 0 (~1/4 of its instructions on one wave; a tree is B class, more accurate,
  estimated +0.3-0.5% end to end); (3) per-head normalisation of wu (+10% at H = 64 with its P-95 diagonal order).

## 75. bwd_tail tree sum (P-938t2, B class, local commit) and the prep_a gate / weighting ceilings
- a02c016 pushed to the fork (f7f2170..a02c016).
- **P-938t2**: bwd_tail's per-CTA dA_log partial (512 fp32 terms in LDS) was one lane's 512-add dependent chain on
  wave 0 (~1/4 of the instructions bwd_tail executes on the production route).  Now wave 0 sums 64 lane runs of 8
  (column-strided) and a fixed shfl-xor tree (a + b == b + a: all lanes agree; deterministic).  gfx938 only,
  KDA_B2_TREE=0 restores the chain.
  - Bitwise vs a02c016: 75 of 84 tensors identical; the 9 dA_log differ (as intended).
  - FP64 oracle (13 cases: 4096/12 x 10 seeds, 8192/48, 16384/12, 32768/4): both deterministic; every tensor but
    A_log identical; A_log error ratio 0.9999-1.0001 (geomean 1.00001, 11 of 13 marginally above).  The summation
    order is not where dA_log's error comes from (bf16 dg / the gate chain dominate), so "a tree is more accurate"
    does not show; the change is neutral to 1e-4 relative.  Rule 4 (Megatron loss) not run: a 1e-5 relative change
    of one reduced gradient is far below that setup's run-to-run spread (section 69).
  - bwd_tail 1.09-1.15x (ABBA, 6 shapes), 7-47 us, +0.4-0.7% end to end.
- prep_a ceilings for a B-class rewrite of the gate and the weighting (timing only, kernel time, ABBA vs a02c016):
| removed entirely | prep_a | end to end at 8192/64 |
|---|---|---|
| cs: the sigmoid (B1 path) | 1.03-1.06x | <= 0.4% |
| cg: the Kahan compensation of the gate scan | 1.01-1.03x | <= 0.3% |
| ce: every exp2 of QgA / Kb / Kn, P1a, P1b | 1.09-1.11x | <= 1.0% |
| all three | 1.13-1.15x | <= 1.1% |
  Even free gate and weighting arithmetic buys at most 13-15% of prep_a (about 1% end to end); any real rewrite gets
  a fraction of that.  **Not pursued.**  The rest of prep_a is loads / stores, the LDS exchange, conversions and MMAC,
  and at H = 64 prep_a is not VALU bound at all (section 72).
- Note: with 5 GPUs measuring at once, kbench's per-iteration totals drift between processes (the same a02c016
  build read 11.1-12.4 ms at 8192/64 on different GPUs); kernel times stay consistent.  End-to-end effects of
  parallel A/Bs are therefore computed as kernel time saved / the clean single-GPU total (section 74).

## 76. Forward split: whole-chain ceiling (in progress)
- d563aa6 pushed (a02c016..d563aa6).
- Cards re-collected on d563aa6 (results/kda-gfx938-cards-20261030-t2), forward kernels only:
| shape | forward t | card floor | t / floor | floors (us) |
|---|---:|---:|---:|---|
| 8192/64 | 3255 | 2666 | 1.22 | prep_a VALU 946, prep_b TA 688, fwd_h TA 1032 |
| 2048/16 | 259 | 183 | 1.42 | prep_a VALU 59, prep_b TA 43, fwd_h TA 81 |
| 16384/8 | 979 | 795 | 1.23 | prep_a VALU 235, prep_b TA 164, fwd_h (16-col) TA 395 |
| 32768/16 | 2958 | 2656 | 1.11 | prep_a VALU 942, prep_b TA 635, fwd_h TA 1080 |
  Inside the current split, 11-29% of the forward is left above the card floors.
- Hand-off-free ceiling ("hfree", timing only): one runtime-zero flag gates every hand-off access between the three
  forward kernels (prep_a stores QgA / Kn / Kb / R / Gn; prep_b loads R / Kb and stores AqkA / Wn / Un; fwd_h loads
  Wn / Kn / QgA / AqkA / Un / Gn); compute and the required inputs / outputs stay.  It bounds every re-split.
  - First build void: the flag was a `volatile` global, i.e. a vector load plus a wait in front of every gated access
    (forward 0.3-0.95x).  Rebuilt with a `__constant__` flag read by readfirstlane (uniform scalar branch).
  - Measured on a quiet nmz5 (ABBA, warm-up, one GPU per shape pair; kernel time, t2 -> hfree):
| shape | prep_a | prep_b | fwd_h | forward saved | of fwd+bwd |
|---|---|---|---|---|---|
| 2048/16 | 1.18x | 1.40x | 1.18x | 48 us (18%) | 5% |
| 2048/64 | 1.23x | 1.67x | 1.55x | 267 us (30%) | 9% |
| 4096/32 | 1.21x | 1.58x | 1.46x | 230 us (27%) | 8% |
| 4096/8 | 1.18x | 1.40x | 0.97x | 29 us (10%) | 3% |
| 8192/64 | 1.24x | **1.80x** | 1.46x | **976 us (30%)** | 8% |
| 8192/16 | 1.22x | 1.59x | 1.00x | 129 us (16%) | 4% |
| 16384/32 | 1.21x | 1.74x | 1.36x | 838 us (27%) | 7% |
| 16384/8 | 1.20x | 1.58x | 0.95x | 99 us (10%) | 3% |
| 32768/64 | 1.22x | 1.64x | 1.26x | 2762 us (24%) | 6% |
| 32768/16 | 1.21x | 1.69x | 0.94x | 436 us (15%) | 4% |
  Reading: (1) a free hand-off would make the gfx938 forward faster than H20's at H = 64 (8192/64: 2265 vs 2445 us);
  the forward gap is the hand-off traffic, not the arithmetic.  (2) prep_b is 40-45% hand-off (R / Kb loads, AqkA /
  Wn / Un stores), prep_a 15-20% (its stores), fwd_h 26-35% at H >= 32 and ~0 at H <= 16 (there fwd_h is chain
  bound with 32-64 CTAs).  (3) The whole re-split ceiling is 3-9% of fwd+bwd; any real split captures part of it.
- First capture, **P-938wd (commit 06e8f2c, pushed)**: fwd_h's K-side reads are issued by all 4 V-slab CTAs of a head.  The
  wide variant (2 CTAs per head, P-49, bit-identical) was gated to NT >= 512 from gfx936 data and only checked on
  gfx938 at 2048/16 and 4096/32.  At H = 64 it wins at every NT: fwd_h 1.04-1.16x (knob sweep), 1.09-1.15x (ABBA),
  fwd+bwd 1.016-1.032x; at H = 32 (64 CTAs) 0.74-0.79x.  Rule: gfx938, H >= 64 -> wide at every NT.  Gate 84/84.
- Lesson (R65): a gate measured on one card / one shape range is a hypothesis on the next; and a deletion ceiling
  that names a kernel's hand-off reads (here 30-55% of fwd_h) points at the existing variants first.

## 77. Re-split option A (w / u computed inside fwd_h): ceiling -> not pursued
Option A: prep_b stops after Aqk / solve / Akk (no pass 0 / 1: no Kb / v reads, no Wn / Un stores); fwd_h computes
w = Ainv . Kb and its u slab = Ainv . (v beta) per chunk from an Ainv tile (8 KB per chunk-head) plus the Kb / v tiles
it would read instead of Wn / Un.  Timing-only build "pa" on 06e8f2c, deliberately **optimistic** for A:
- prep_b side: the existing G2_PREPB_STOP=3 (returns after solve + Akk).
- fwd_h side (wide kernel, the H >= 64 and H <= 8 routes): +8 KB Ainv load per CTA per step from distinct memory,
  +32 MMAC (w, 2 K-tiles x 4 row blocks x 4 j) + 4 x NVH MMAC (u) per wave per step, results folded into the next
  step's W / u registers (so GEMM1 waits for them).  Not modelled (all extra cost for a real A): redistributing w
  across waves through LDS (16 KB per step), transposing v beta into a B operand, prep_a / prep_b writing Kb^T and
  Ainv in operand layouts.  VGPR (MODE 5): NVH 4 214 -> 228, NVH 1 148 -> 178; no scratch.
ABBA, kernel time (one build, env A vs B; GPU 1-3 in parallel):
| shape | prep_b saved (STOP=3) | fwd_h added (plusA) | net forward | fwd+bwd (combined run) |
|---|---|---|---|---|
| 2048/64 | -100 us (1.90x) | +20 us (0.95x) | -80 us | 1.027x |
| 8192/64 | -385 us (1.96x) | +154 us (0.88x) | -231 us | 1.021x |
| 16384/64 | -535 us | +407 us (0.84x) | -128 us | 1.006x |
| 4096/8 | -26 us | +94 us (0.58x) | +68 us | **0.938x** |
| 16384/8 | -87 us | +370 us (0.57x) | +283 us | **0.923x** |
- At H <= 8 fwd_h is chain bound: the added MMAC sits on the serial step.  At H = 16 / 32 fwd_h runs the 4-slab kernel
  (wide loses 0.74-0.79x there), so w would be recomputed 4x.  At H = 64 the optimistic gain is 2-2.6% and the
  unmodelled LDS exchange / layout writes come out of it.  **Not built.**
- What the ceiling does show: prep_b's pass 0 / 1 (Kb -> Wn, v beta -> Un) are ~49% of prep_b at H = 64; with section
  76, the hand-off cost of the gfx938 forward sits in the stores (TA write ~2.7x the read cost per byte) of the
  three-kernel structure, and no re-split tried here (fusion: residency; A: redundant w + chain) removes them cheaper.

## 78. Store cost per access form on gfx938 (examples/kda/gfx938/st_forms.hip) and P-938aq
8192 CTAs x 256 threads, 512 MB of useful bytes per launch, residency 2 / 3 / wave-limited (dynamic LDS pad):
| form | GB/s | TA busy (M cycles) | write requests |
|---|---:|---:|---:|
| 16 / 8 / 4 B per lane, contiguous (1 KB / 512 / 256 B per wave store) | 1660-1850 | 25.6 | 8.39M (64 B each) |
| 16 B per lane, 256 B row segments, row stride 16 KB | 1660-1770 | 25.9 | 8.39M |
| 16 B per lane, 128 B row segments, row stride 8 KB | 1240-1280 | 37.9 | 8.39M |
| 8 B per lane, 64 B row segments | 940-1010 | 48.7 | 8.39M |
| 8 B per lane, 32 B row segments, 4 consecutive stores complete each 128 B row (prep_b Aqk / Akk, fwd_h o) | **730** | 68.6 | 16.78M (32 B each) |
| 2 B per lane, scattered | 1160-1310 | 38.0 | 8.39M |
| nontemporal variants | no gain | -- | -- |
- A store costs per 64 B write request (TA ~3 cycles each) and per line it touches; writing whole lines (or >= 256 B
  row segments) runs at the HBM write peak.  32 B row pieces cost 2.5x per byte.
- Production write requests per call (PMC TCP_TCC_WRITE_REQ_sum, 8192/64) vs written bytes: prep_a 63 B / request,
  dhu / dav / gate 64, bwd_tail 44-52, fwd_h (wide) 48, s4b 47, prep_b 45, s4a 39, wu 36.
- **P-938aq** (prep_b: Aqk rows held until after the solve, Aqk + Akk staged per wave in the dead solve tiles and
  emitted as whole 128 B rows, 8 lanes x 16 B): bit-identical (84/84), but prep_b 0.96-1.01x (ABBA, 6 shapes).  The
  32 B-piece stores were overlapped with the solve; moving them behind a barrier as full rows bought nothing.
  Reverted.  Rule (R66): a store form that is slow in isolation is a candidate only if a deletion ceiling shows the
  store on the critical path (fwd_h's o: cB2 1.13-1.16x, section 66; prep_b's Aqk / Akk: not established).
