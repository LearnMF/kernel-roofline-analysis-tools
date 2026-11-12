# M2c: handoff pricing and VGPR estimate

Follow-up to `m2b-split-search.md`.  M2b found that the generic search over-charged the production split
(4183 us vs 3710 us for the same p54a grouping) and under-estimated VGPR (0.4-0.75x of the measured
kernels).  Both were fixed from semantics only; the measured `CORE` footprints and the production timings
are still validation-only (a test enforces that `kda_partition` is not imported by the predictor).

## 1. What the over-charge actually was: unversioned dataflow, not layouts

Comparing the search's per-kernel HBM bytes with the optrace-validated hand-written IR kernel by kernel
showed the gap sat in one place: `fwd_prep` read 403 MB instead of 202 MB.  The backward recompute
re-produces `qn`, `kn`, `W`, `U`, ... under the same tensor name, and the M2b pricing treated "a tensor
some *other* kernel produces" as a read, so the forward kernel was charged for reading back the
backward's recomputed copy.

Fix (`kra/model/capacity.py::dataflow`): versioned def-use.  Every read is bound to the latest producer
before it; a value is written to HBM only if a consumer of *that version* is outside the kernel.  This is
an operator-independent correctness fix, not a KDA rule.

After it, for every production kernel the search's bytes are **at or below** the hand-written IR:

| kernel | search rd / wr MB | hand IR rd / wr MB | difference |
|---|---:|---:|---|
| fwd_prep | 202.1 / 254.8 | 202.1 / 280.0 | hand writes a second layout of Kg/Qg |
| wu | 277.6 / 251.7 | 277.6 / 352.3 | hand writes W in natA *and* natB, U in natC |
| dav | 125.8 / 100.7 | 125.8 / 151.0 | hand writes dVn (natC copy) |
| head_gate | 50.3 / 102.2 | 50.3 / 100.7 | g fp32 vs Gn vecK (dtype rule) |
| others | equal | equal | |

The remaining differences are exactly the **layout copies** (`Mat` with a non-row layout or a `tag`).
They are deliberately **not** charged by the search: a consumer can re-fragment a row-major tile on load
through LDS, so a second HBM layout is an implementation choice (it trades HBM bytes for LDS shuffles),
not a requirement.  Charging them would make `T_split` stop being a lower bound and would require knowing
which layouts the production kernels chose -- i.e. the answer.  The per-op IR does not say which operand
of a product is A or B, so there is no semantic basis for deriving the copies either.  The hand-written
decomposition keeps them, so `T_split(search) <= T_split(hand IR)` holds by construction and is tested.

## 2. Handoffs that do not fit on chip: spill inside the kernel

M2b could only say "s4a+s4b does not fit (80 KB LDS)".  A kernel writer has a third option besides
on-chip and separate kernels: keep one launch, write the handoff to global memory in phase 1 and read it
back in phase 2.  The shared inputs are still loaded once.  `footprint(..., lds_cap=...)` now does this:
when the LDS peak passes the card's `lds_per_cu`, the largest LDS-resident handoff is spilled (repeat
until it fits); `kernel_work(..., spilled=...)` charges one write + one read for it.

Without being told, the search spills `dA@wy_bwd_a` and `dwT@wy_bwd_a` in the best gfx936 split -- which
is the production `wy_s4_split` structure ("s4a writes dA (fp32) + dwT, s4b reads them").  With the spill
rule, no grouping is infeasible on 64 KB any more; on a 24 KB card some single ops no longer fit and are
pruned (tested).

## 3. VGPR: operands, addressing allowance, register-to-LDS fallback

M2b counted accumulators only.  Added, from semantics:

| term | rule | provenance / scope |
|---|---|---|
| accumulators | output tiles, fp32 for products | unchanged |
| operands | a product op holds its non-staged operands; one product has two operands, so the two largest non-staged input tiles | algorithmic, any MMAC op |
| addressing / loop state | `REG_BASE_VGPR = 16` per thread per kernel | **heuristic assumption about the compiler**, not measured, not fitted; `--vgpr acc` reports without it |
| register pressure | operands go to registers first; past `max_vgpr_per_thread` they move to LDS (largest first) | what a kernel writer does; reported as `operands_moved_to_lds` |

Comparison with the measured production kernels (validation only):

| op | VGPR acc-only (M2b) | VGPR full (M2c) | measured |
|---|---:|---:|---:|
| re_l2norm | 32 | 48 | 20 |
| re_gate | 33 | 49 | 96 |
| re_wy | 160 | 208 | 212 |
| dav | 48 | 88 | 76 |
| wy_bwd_a | 80 | 128 | 152 |
| wy_bwd_b | 96 | 176 | 236 |
| tail | 48 | 64 | 120 |

The MMAC ops move from 0.4-0.75x to 0.75-1.16x of the measured footprint.  `re_gate` and `tail` stay
low: their scratch (cumulative-sum buffers, per-row reduction partials) is not expressed in the IR, and
inventing it from the measured number would be fitting.  `re_l2norm` is over-estimated (lane-local op,
the measured kernel keeps far less than one row tile live).

## 4. Results after M2c

16K / H=12, bf16, fwd+bwd:

| | M2b | M2c |
|---|---:|---:|
| gfx936 best `T_split` | 3.295 ms (keep_state / W-free) | **2.675 ms** (keep_state / dAqk in s4b, 6 kernels) |
| gfx938 best `T_split` | 2.113 ms (dAqk in s4b / keep_state) | **1.710 ms** (same formulation and split as gfx936) |
| production p54a grouping, priced by the search | 4183 us (rank 260 / 512) | **3672 us** (rank 412 / 1024) |
| production p54a, hand-written IR | 3710 us | 3710 us |
| `T_theory` (fully fused) | 0.837 ms | 0.837 ms |

H=48: gfx936 10.70 ms, gfx938 6.84 ms, same formulation and split.  In M2b the two cards picked different
formulations; now both agree (the card changes the time, not the split).  Three things changed at once
(versioned dataflow, spill, VGPR operands) and which one flipped gfx936 away from W-free is not isolated.  gfx938 still has no
`recurrence_step_us`, so its serial bound is omitted, not borrowed.

What the hold-out now says:

* **absolute**: the generic search prices the production grouping within 1% of the optrace-validated
  hand IR (3672 vs 3710 us; the 38 us are the layout copies).  This is the claim M2b could not make.
* **ranking**: the production grouping is in the middle of the list, and that is structural, not noise.
  `T_split` is a lower bound in which fusing never costs anything once it fits; the minimum per number of
  kernels falls monotonically (8 kernels 3028 us ... 18 kernels 4358 us).  Production keeps more kernels
  for reasons a lower bound does not price -- occupancy (the best split runs at 256 VGPR, 1 tile/CU),
  latency hiding, compile-time register allocation.  So "rank of the production split" is not a useful
  acceptance metric for a lower-bound tool; "the search prices a given split like the validated IR does"
  and "the search finds the same sub-kernel boundary and spills" are.

## Known gaps (updated)

* occupancy is reported but not priced -- now the main reason the search prefers fewer, fatter kernels;
* layout copies are intentionally not charged to HBM; their LDS shuffle traffic is not charged either;
* VGPR for lane-local ops with scratch (`re_gate`, `tail`) is still low; `REG_BASE_VGPR` is a heuristic;
* the operand rule uses the two largest non-staged inputs because the IR does not pair operands.
