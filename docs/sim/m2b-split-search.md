# M2b: capacity-constrained kernel split search (shape + machine only)

`kra predict --split-search` solves the part of M2 that was left open: given a shape and a machine file,
which kernels the operator has to be cut into, and what that split costs.

```
python -m kra predict --machine machines/gfx936-bw1000.json --op kda \
    --shape T=16384,H=12 --phase fwd+bwd --split-search
```

## What is searched

`kra/model/capacity.py` is operator-independent.  A plugin hands it a *capacity graph* -- its ops, the
tensors between them, and the on-chip tile each op needs -- and the solver

1. cuts the op list at **barriers** (an op whose parallel structure differs from its neighbours: a scan
   among chunk-parallel ops, a reduction over the sequence).  A barrier cannot share a kernel: the grid
   is different;
2. enumerates every grouping of **contiguous** ops between barriers (1024-2048 candidates for KDA
   fwd+bwd, depending on the formulation);
3. drops the groupings whose working set exceeds `device.lds_per_cu` or 256 VGPR;
4. prices the survivors with the same bound model as `T_theory` (each kernel reads what it does not
   produce and writes what another kernel needs) and ranks them by `T_split`;
5. repeats over the **formulation** variants the plugin declares (state recompute vs keep-state,
   W/U materialised vs W-free, dAqk in dav vs in s4b), so a different formulation can win on a
   different machine.

An operator whose plugin exposes no capacity graph gets no split -- the solver says so instead of
inventing one.

## Where the footprints come from (and why not from the measured kernels)

`examples/kda/kda_partition.py` prunes with the **measured** LDS/VGPR of today's production kernels.
Searching with those is fitting the answer: they only exist for KDA on gfx936, after the optimisation.
The solver therefore derives the working set from the IR:

* an op that contracts over the token axis (`tri`/`sq`/`t3` products) must make one operand tile visible
  to all lanes -> its largest input tile is staged in LDS; a triangular inverse also keeps its C x C
  matrix there across the sweep;
* accumulators are the output tiles, in fp32 for anything a product accumulates;
* a tensor handed from one op to another inside a kernel stays live in between: a per-token row tile keeps
  the lane mapping (registers), a C x C / state / K x C tile goes through LDS.

Cross-check against the measured footprints (validation only -- these numbers are not inputs):

| op | analytic LDS KB | measured LDS KB | analytic VGPR | measured VGPR |
|---|---:|---:|---:|---:|
| re_l2norm | 0.0 | 0.0 | 32 | 20 |
| re_gate | 0.0 | 2.0 | 33 | 96 |
| re_wy | 32.0 | 26.2 | 160 | 212 |
| dav | 16.0 | 17.0 | 48 | 76 |
| wy_bwd_a | 48.0 | 29.8 | 80 | 152 |
| wy_bwd_b | 32.0 | 32.0 | 96 | 236 |
| tail | 0.0 | 6.0 | 48 | 120 |

LDS lands within 0.9-1.6x where the op stages anything, and misses the small scratch buffers of the
ops that stage nothing (`re_gate`, `tail`).  VGPR is systematically **low** (0.4-0.75x): the model counts
accumulators only, not address registers, prefetch buffers or the compiler's spill margin.  A low VGPR
estimate is optimistic -- it can call a grouping feasible that a real kernel cannot reach -- so the
feasibility verdict for VGPR is weaker than the one for LDS.  The two residency rules (`--residency
shape` vs `--residency lds`, the latter charging every handoff to LDS) bracket the LDS side.

## Results, 16K / H=12, bf16, fwd+bwd

| formulation | candidates | feasible | best T_split | worst feasible |
|---|---:|---:|---:|---:|
| state=recompute | 1024 | 512 | 3608 us | 4359 us |
| state=keep_state | 2048 | 1024 | 3383 us | 4135 us |
| state=recompute / wy=wfree | 1024 | 512 | 3503 us | 4273 us |
| **state=keep_state / wy=wfree** | 2048 | 1024 | **3295 us** | 4065 us |
| dAqk=dAqk_s4b / state=recompute | 1024 | 512 | 3524 us | 4276 us |
| dAqk=dAqk_s4b / state=keep_state | 2048 | 1024 | 3300 us | 4051 us |

`T_theory` (fully fused) is 837 us, so every split pays 3.9-5.2x in interface traffic.

**The search picks keep-state on its own.**  Nothing tells it that P-82 (the forward saving the state) was
the round that won on gfx936; keep-state wins because it removes the recompute scan and its interface
tensors.  That is the one result here that is not a restatement of an input.

## Hold-out: where does the production split rank?

In the `state=recompute` formulation (the 10-13 production route), the p54a-like grouping ranks
**260 of 512** feasible candidates at 4183 us against the best 3608 us.  Two reasons, both worth stating
rather than tuning away:

* the production route is p54a **plus** the layout machinery the generic model does not know: it reads and
  writes specific materialisations (natA/natB/natC fragments, bN state blocks), and several of its
  crossings are cheaper than the full tensor the search charges.  The hand-written `T_split` for the same
  split is 3710 us, not 4183 us.  **Absolute T_split from the search is more conservative than the
  layout-aware hand-written decomposition; only the ranking inside one search is comparable.**
* `wy_bwd_a+wy_bwd_b` fused is **infeasible** in the model (80 KB LDS, the `dA` and `dwT` handoffs on top
  of s4a's own staging).  Production agrees: its `wy_s4_split` launcher runs them as two sub-kernels with
  `dA`/`dwT` going through HBM in between.  The capacity term reproduces a decision that was made by hand.

So the honest reading is: the solver finds the capacity frontier and the formulation choice, and it
reproduces the production split's sub-kernel boundary, but it does not yet model layouts, so it cannot
rank the production route against its neighbours fairly.  Layout-aware materialisation is the next step.

## Cross-machine check (gfx938 / BW1102)

Same operator, same shape, only the machine file changes (72 CU at 1.3 GHz, 256-bit HBM at 875 MHz
-> 2015 GB/s read vs 1339, 371 TFLOPS bf16 vs 476, same 64 KB LDS):

| machine | best formulation | best T_split | kernels |
|---|---|---:|---:|
| gfx936-bw1000 | state=keep_state / wy=wfree | 3295 us | 7 |
| gfx938-bw1102 | dAqk=dAqk_s4b / state=keep_state | 2113 us | 7 |

The winning formulation changes with the machine, which is the behaviour a what-if tool needs; the
grouping stays the same because on both cards every kernel is HBM-bound, so the search is choosing the
formulation with the least interface traffic.  Note gfx938's machine file has no `structural` block, so
on it the recurrence step latency is unknown: the report says the serial chain is **not** in the bound
instead of substituting the gfx936 numbers.

## Multi-level roofline

`predict` now reports the operator's position against every level the machine file has a roof for, and
charges the level as a bound term when the operator can bound its traffic from below (every HBM byte also
crosses L2; staged tiles must be written to and read from LDS).  For KDA 16K/H=12 fwd+bwd on gfx936:

| level | attainable bw | required bytes | intensity | ridge | bytes/bw |
|---|---:|---:|---:|---:|---:|
| HBM | 1339 GB/s | 810.8 MB | 122.2 | 355.4 | 605.4 us |
| L2 | 6101 GB/s | 810.8 MB | 122.2 | 78.0 | 132.9 us |

The operator is memory-bound against HBM and compute-bound against L2, which is the quantitative version
of "the fully fused operator would be compute-bound if the traffic stayed on chip".

## Known gaps

* layouts / materialisations are not modelled, so split traffic is charged at full tensor bytes (above);
* VGPR is underestimated (above), making the VGPR feasibility verdict optimistic;
* occupancy is reported (`tiles/CU`) but not priced: a 1-tile-per-CU kernel is not slowed down in the bound;
* the search ranks by `T_split` only; it does not consider register pressure effects on instruction-level
  parallelism, which is a `T_full` concern.
