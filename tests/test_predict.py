from pathlib import Path

import pytest

from kra.model.predict import TheoryMachine, predict
from kra.ops import get

ROOT = Path(__file__).resolve().parent.parent
BW1000 = ROOT / "machines" / "gfx936-bw1000.json"
PROFILE = ROOT / "examples" / "bw1000_profile" / "bw1000_torch271_dtk2604.yaml"


@pytest.fixture(scope="module")
def mach():
    return TheoryMachine.load(str(BW1000))


def test_gemm_work_is_textbook():
    w = get("gemm").work({"M": 8192, "N": 4096, "K": 6144}, "bf16", "fwd")
    assert w.flops == 2 * 8192 * 4096 * 6144
    assert w.bytes_min == (8192 * 6144 + 6144 * 4096 + 8192 * 4096) * 2
    wb = get("gemm").work({"M": 8192, "N": 4096, "K": 6144}, "bf16", "fwd+bwd")
    assert wb.flops == 3 * w.flops and wb.kernels == 3 and len(wb.parts) == 3


def test_large_gemm_is_compute_bound_at_peak(mach):
    r = predict(mach, "gemm", {"M": 8192, "N": 8192, "K": 8192}, "bf16", "fwd")
    assert r["bottleneck"] == "compute"
    assert r["T_theory_us"] == pytest.approx(2 * 8192 ** 3 / 475.955e12 * 1e6)
    assert r["arithmetic_intensity_flop_per_byte"] > r["ridge_point_flop_per_byte"]


def test_gemv_is_memory_bound(mach):
    r = predict(mach, "gemm", {"M": 2, "N": 6144, "K": 12288}, "bf16", "fwd")
    assert r["bottleneck"] == "memory"
    assert r["ceiling"]["bw_util_pct"] == pytest.approx(100.0)


def test_tiny_shape_hits_launch_floor(mach):
    r = predict(mach, "gemm", {"M": 1, "N": 64, "K": 64}, "bf16", "fwd")
    assert r["bottleneck"] == "fixed"
    assert r["T_theory_us"] == pytest.approx(mach.launch_us)


def test_theoretical_peaks_give_a_lower_bound(mach):
    th = TheoryMachine.load(str(BW1000), "theoretical")
    s = {"M": 4096, "N": 4096, "K": 4096}
    assert predict(th, "gemm", s, "bf16", "fwd")["T_theory_us"] <= predict(mach, "gemm", s, "bf16", "fwd")["T_theory_us"]


def test_rmsnorm_is_memory_bound(mach):
    r = predict(mach, "rmsnorm", {"elements": 1 << 24}, "bf16", "fwd")
    assert r["bottleneck"] == "memory"
    assert r["T_theory_us"] == pytest.approx(2 * 2 * (1 << 24) / mach.bw * 1e6)


def test_unknown_op_is_refused():
    with pytest.raises(SystemExit, match="no plugin"):
        get("dsa")


def test_missing_shape_key_is_refused(mach):
    with pytest.raises(SystemExit, match="needs --shape"):
        predict(mach, "gemm", {"M": 1, "N": 2}, "bf16", "fwd")


def test_bw1000_profile_bound_holds_and_beats_tiers(mach):
    yaml = pytest.importorskip("yaml")
    from kra.model.validate import gemm_rows, non_gemm_rows, summarize
    prof = yaml.safe_load(PROFILE.read_text())
    gs = summarize(gemm_rows(mach, prof))
    ns = summarize(non_gemm_rows(mach, prof))
    assert gs["used"] >= 100 and not gs["bound_violations"]
    assert not ns["bound_violations"]
    p = gs["predictor"]
    kra = next(v for k, v in p.items() if k.startswith("kra: floor"))
    tier = next(v for k, v in p.items() if k.startswith("hcu_tiered"))
    assert kra["mape_pct"] < 20.0
    assert kra["mape_pct"] < tier["mape_pct"]


# KDA reference: docs/kda_report.md 3.1 (16K/TP8 = T 16384 H 12; TP2 = H 48), bounds in ms
KDA_REF = {12: {"theory": 0.83, "p54a": 3.70, "p82": 3.35}, 48: {"theory": 2.62, "p54a": 14.38, "p82": 13.37}}


@pytest.mark.parametrize("H", [12, 48])
def test_kda_four_layer_bounds_from_shape_only(mach, H):
    r = predict(mach, "kda", {"T": 16384, "H": H}, "bf16", "fwd+bwd")
    ref = KDA_REF[H]
    assert r["T_theory_us"] / 1e3 == pytest.approx(ref["theory"], rel=0.02)
    for name in ("p54a", "p82"):
        assert r["T_split"][name]["T_split_us"] / 1e3 == pytest.approx(ref[name], rel=0.01)
    ts = [r["T_split"][n]["T_split_us"] for n in ("p54a", "p77b", "p82")]
    assert r["T_theory_us"] < ts[2] <= ts[1] <= ts[0]
    assert r["bottleneck"] == ("serial" if H == 12 else "memory")


def test_kda_interface_bytes_match_ledger(mach):
    r = predict(mach, "kda", {"T": 16384, "H": 12}, "bf16", "fwd+bwd")
    assert r["T_split"]["p82"]["bytes"] / r["work"]["bytes_min"] == pytest.approx(5.08, abs=0.01)


def test_kda_ir_matches_optrace():
    from kra.ops import kda_ir
    if not Path(kda_ir.CARDS).exists():
        pytest.skip("ceiling cards not in checkout")
    worst, _ = kda_ir.validate()
    assert worst < 1e-6


def test_chain_falls_back_to_the_generic_floor_then_drops_out():
    """Without a calibrated step the chain uses the measured dependent-MMAC + barrier floor, which is
    looser; without those latencies either it is declared absent rather than guessed."""
    import json
    m = json.loads(BW1000.read_text())
    m.pop("structural")
    r = predict(TheoryMachine(m), "kda", {"T": 16384, "H": 12}, "bf16", "fwd+bwd")
    assert r["bottleneck"] != "serial"
    assert any("generic floor" in n for n in r["notes"])
    full = predict(TheoryMachine(json.loads(BW1000.read_text())), "kda", {"T": 16384, "H": 12},
                   "bf16", "fwd+bwd")
    assert full["T_theory_us"] >= r["T_theory_us"]
    m.pop("latency")
    r2 = predict(TheoryMachine(m), "kda", {"T": 16384, "H": 12}, "bf16", "fwd+bwd")
    assert any("serial chain is NOT" in n for n in r2["notes"])


# ------------------------------------------------------------------ M2b: capacity split search
def _m936():
    from kra.model.predict import TheoryMachine
    return TheoryMachine.load("machines/gfx936-bw1000.json")


def test_capacity_barriers_cannot_be_grouped():
    from kra.model.capacity import segments
    from kra.ops.kda import capacity_graph
    g = capacity_graph({"T": 16384, "H": 12})
    segs = segments(g)
    for kind, ops in segs:
        if kind == "barrier":
            assert len(ops) == 1 and ops[0].barrier
    assert any(kind == "barrier" for kind, _ in segs)


def test_capacity_search_prunes_on_lds_and_prices_the_rest():
    import copy
    from kra.model.capacity import search
    from kra.ops.kda import capacity_graph
    m = _m936()
    g = capacity_graph({"T": 16384, "H": 12})
    r = search(g, m, "mmac_bf16", m.peaks["mmac_bf16"])
    ts = [x["T_split_us"] for x in r["ranked"]]
    assert ts == sorted(ts) and ts[0] < ts[-1]
    assert all(x["lds_peak_kb"] <= 64.0 for x in r["ranked"])
    assert all(x["vgpr_peak"] <= 256 for x in r["ranked"])
    assert any(x["spilled"] for x in r["ranked"])            # 64 KB binds: some fusions need a spill
    small = copy.deepcopy(m)
    small.device["lds_per_cu"] = 24 * 1024                  # an op's own staging no longer fits
    rs = search(capacity_graph({"T": 16384, "H": 12}), small, "mmac_bf16", m.peaks["mmac_bf16"])
    assert rs["feasible"] < rs["candidates"]


def test_fusing_s4a_s4b_cannot_stay_on_chip_and_spills_a_handoff():
    # production found the same: s4a and s4b share one launch but pass dA / dwT through global memory
    from kra.model.capacity import footprint, kernel_work
    from kra.ops.kda import capacity_graph
    g = capacity_graph({"T": 16384, "H": 12})
    byn = {o.name: o for o in g.ops}
    k = [byn["wy_bwd_a"], byn["wy_bwd_b"]]
    assert footprint(g, k)["lds_bytes"] > 65536
    fp = footprint(g, k, lds_cap=65536)
    assert fp["lds_bytes"] <= 65536 and fp["handoffs_spilled"]
    on, sp = kernel_work(g, k), kernel_work(g, k, spilled=fp["handoffs_spilled"])
    assert sp.bytes_rd > on.bytes_rd and sp.bytes_wr > on.bytes_wr
    a, b = kernel_work(g, k[:1]), kernel_work(g, k[1:])
    assert sp.bytes_rd < a.bytes_rd + b.bytes_rd               # the shared inputs are still read once


def test_split_search_is_bounded_below_by_t_theory():
    from kra.model.capacity import solve
    from kra.model.predict import predict
    m = _m936()
    shape = {"T": 16384, "H": 12}
    t_theory = predict(m, "kda", shape, "bf16", "fwd+bwd")["T_theory_us"]
    s = solve(m, "kda", shape, "bf16", "fwd+bwd")
    assert s["best"]["T_split_us"] > t_theory
    for f in s["formulations"]:
        if "T_split_us" in f:
            assert f["T_split_us"] >= s["best"]["T_split_us"]


def test_split_search_changes_with_the_machine():
    from kra.model.capacity import solve
    from kra.model.predict import TheoryMachine
    shape = {"T": 16384, "H": 12}
    a = solve(_m936(), "kda", shape, "bf16", "fwd+bwd")
    b = solve(TheoryMachine.load("machines/gfx938-bw1102.json"), "kda", shape, "bf16", "fwd+bwd")
    assert a["best"]["T_split_us"] != b["best"]["T_split_us"]


def test_conservative_residency_prunes_more_and_is_never_faster():
    from kra.model.capacity import solve
    m = _m936()
    shape = {"T": 16384, "H": 12}
    a = solve(m, "kda", shape, "bf16", "fwd+bwd", residency_mode="shape")
    b = solve(m, "kda", shape, "bf16", "fwd+bwd", residency_mode="lds")
    assert b["best"]["T_split_us"] >= a["best"]["T_split_us"]


def test_operator_without_capacity_graph_is_refused():
    import pytest
    from kra.model.capacity import solve
    with pytest.raises(SystemExit):
        solve(_m936(), "gemm", {"M": 4096, "N": 4096, "K": 4096}, "bf16", "fwd")


def test_hierarchical_roofline_levels_are_reported():
    from kra.model.predict import predict
    r = predict(_m936(), "kda", {"T": 16384, "H": 12}, "bf16", "fwd+bwd")
    assert "hbm" in r["levels"] and "l2" in r["levels"]
    assert r["levels"]["hbm"]["bound"] == "memory"
    assert r["levels"]["l2"]["ridge_flop_per_byte"] < r["levels"]["hbm"]["ridge_flop_per_byte"]


def test_measured_production_footprints_are_not_an_input_of_the_search():
    src = open("kra/model/capacity.py", encoding="utf-8").read() + \
        open("kra/ops/kda.py", encoding="utf-8").read()
    assert "kda_partition" not in src        # the measured CORE table stays out of the predictor


def test_split_search_handoff_bytes_match_the_optrace_validated_ir():
    # versioned dataflow: a recomputed tensor is a new value, so the forward kernel does not read back what
    # it produced itself.  With that, the search's bytes per production kernel never exceed the
    # optrace-validated hand-written decomposition (which also pays extra layout copies).
    from kra.model import capacity as cap
    from kra.ops.kda import capacity_graph
    import kra.ops.kda_ir as ir
    T, H = 16384, 12
    g = capacity_graph({"T": T, "H": H}, formulation={"state": "recompute"})
    byn = {o.name: o for o in g.ops}
    for k in ir.PRODUCTION:
        w = cap.kernel_work(g, [byn[n] for n in k.ops if n in byn])
        rd, wr, rdh, wrh = k.io_bytes(64, H)
        assert w.bytes_rd <= (rd * T * H + rdh * H) * 1.001
        assert w.bytes_wr <= (wr * T * H + wrh * H) * 1.001 + 2e6   # head_gate: g in fp32 vs Gn copies


def test_operands_spill_to_lds_when_registers_run_out():
    from kra.model import capacity as cap
    from kra.ops.kda import capacity_graph
    g = capacity_graph({"T": 16384, "H": 12})
    op = next(o for o in g.ops if o.name == "wy_bwd_b")
    roomy = cap.footprint(g, [op], vgpr_max=1024)
    tight = cap.footprint(g, [op], vgpr_max=128)
    assert roomy["operands_moved_to_lds"] == 0
    assert tight["operands_moved_to_lds"] > 0 and tight["lds_bytes"] > roomy["lds_bytes"]
    assert tight["vgpr"] <= 128 + 1e-9 or tight["vgpr"] < roomy["vgpr"]


def test_production_grouping_t_split_matches_the_hand_written_decomposition():
    # hold-out: the generic search prices the production p54a grouping within a few % of the IR's
    # optrace-validated decomposition (which also pays the layout copies the search does not need)
    from kra.model.capacity import search, rank_of
    from kra.model.predict import predict
    from kra.ops.kda import capacity_graph
    m = _m936()
    shape = {"T": 16384, "H": 12}
    r = search(capacity_graph(shape, formulation={"state": "recompute"}), m, "mmac_bf16", m.peaks["mmac_bf16"])
    prod = ["l2norm_q+l2norm_k+beta_sig+gate_cumsum+kkt_qkt+tri_inverse+wy_wu", "scan_fwd_o", "re_l2norm",
            "re_gate", "re_wy", "re_scan", "dav", "scan_dh", "wy_bwd_a+wy_bwd_b", "tail", "param_reduce"]
    k = rank_of(r, prod)
    assert k is not None
    hand = predict(m, "kda", shape, "bf16", "fwd+bwd")["T_split"]["p54a"]["T_split_us"]
    t = r["ranked"][k - 1]["T_split_us"]
    assert t <= hand and t > 0.95 * hand


# ------------------------------------------------------------------ occupancy / concurrency (M2d)
def test_bandwidth_attainable_at_a_cta_count_is_below_the_streaming_peak():
    m = _m936()
    peak = m.peaks["hbm_read"]
    few, half, all_ = (m.bw_at_ctas("hbm_read_vs_ctas", n) for n in (8, 32, 10 ** 6))
    assert few < half < all_ <= peak * 1.001
    assert m.bw_at_ctas("no_such_curve", 8) is None


def test_a_register_heavy_kernel_has_a_smaller_in_flight_window():
    m = _m936()
    light = m.concurrency({"ctas": 10 ** 5, "threads": 256, "lds_bytes": 4096, "vgpr": 32})
    heavy = m.concurrency({"ctas": 10 ** 5, "threads": 256, "lds_bytes": 65536, "vgpr": 250})
    assert light["waves_per_cu"] > heavy["waves_per_cu"]
    assert light["in_flight_bytes_per_cu"] > heavy["in_flight_bytes_per_cu"]
    # a kernel whose registers are all live still gets a floor, flagged as an assumption
    full = m.concurrency({"ctas": 10 ** 5, "threads": 256, "lds_bytes": 65536, "vgpr": 256})
    assert full["in_flight_bytes_per_cu"] > 0 and full["notes"]


def test_occupancy_terms_only_tighten_the_bound():
    # every concurrency term is a lower bound on time: switching them on can only raise T_split
    from kra.model.capacity import solve
    shape = {"T": 16384, "H": 12}
    on = solve(_m936(), "kda", shape, "bf16", "fwd+bwd")["best"]["T_split_us"]
    from kra.model.predict import TheoryMachine as TM
    off = solve(TM.load("machines/gfx936-bw1000.json", occupancy=False), "kda", shape, "bf16",
                "fwd+bwd")["best"]["T_split_us"]
    assert on >= off


def test_with_occupancy_the_cheapest_split_is_not_the_most_fused_one():
    # without concurrency terms a lower bound can always fuse for free, so it ranks the fewest kernels
    # first; charging the CUs and the in-flight window a kernel can occupy puts the optimum inside.
    from kra.model.capacity import search
    from kra.ops.kda import capacity_graph
    r = search(capacity_graph({"T": 16384, "H": 12}, formulation={"state": "recompute"}),
               _m936(), "mmac_bf16", _m936().peaks["mmac_bf16"])
    by_n = {}
    for row in r["ranked"]:
        by_n[row["n_kernels"]] = min(by_n.get(row["n_kernels"], 1e18), row["T_split_us"])
    assert by_n[min(by_n)] > min(by_n.values())
    assert r["ranked"][0]["n_kernels"] > min(by_n)


def test_a_kernel_whose_grid_is_smaller_than_the_device_pays_for_the_idle_cus():
    from kra.ops import Work
    from kra.model.predict import _terms
    m = _m936()
    w = lambda ctas: Work(flops=1e11, bytes_rd=1e7, bytes_wr=0.0, other={"valu_lane": 1e10},
                          occupancy={"ctas": ctas, "threads": 256, "lds_bytes": 8192, "vgpr": 64})
    wide = _terms(m, w(10 ** 5), "mmac_bf16", m.peaks["mmac_bf16"])
    narrow = _terms(m, w(8), "mmac_bf16", m.peaks["mmac_bf16"])
    assert narrow["compute:mmac_bf16"] > 9 * wide["compute:mmac_bf16"]
    assert narrow["other:valu_lane"] > 9 * wide["other:valu_lane"]


# ------------------------------------------------------- M2e: dependency-chain and barrier latency
def test_barrier_latency_is_taken_at_the_nearest_calibrated_block_size():
    m = _m936()
    assert m.barrier_s(256) <= m.barrier_s(1024)
    assert m.barrier_s(10 ** 6) == m.barrier_s(1024)        # clamped, not extrapolated
    import json
    bare = TheoryMachine(json.loads(BW1000.read_text()) | {"latency": {}})
    assert bare.barrier_s(256) is None and bare.dep_step_s(256) is None


def test_the_generic_step_floor_is_below_the_calibrated_step():
    m = _m936()
    floor = m.dep_step_s(256)
    cal = m.step_s("kda_step_with_o", 1)
    assert 0 < floor < cal                                   # so max() keeps the calibrated one


def test_a_card_without_structural_still_bounds_the_serial_chain():
    from kra.model.predict import TheoryMachine as TM
    m = TM.load("machines/gfx938-bw1102.json")
    assert not m.structural
    r = predict(m, "kda", {"T": 16384, "H": 12}, "bf16", "fwd+bwd")
    assert any("generic floor" in n for n in r["notes"])
    assert r["kernels"][0]["terms_us"].get("serial:chain", 0) > 0


def test_barriers_are_charged_per_tile_batch_and_only_when_derived():
    from kra.ops import Work
    from kra.model.predict import _terms
    m = _m936()
    occ = {"ctas": 80 * 4, "threads": 256, "lds_bytes": 8192, "vgpr": 64}
    w = lambda n: Work(flops=1e10, bytes_rd=1e7, bytes_wr=0.0, occupancy=occ,
                       barriers=(n, 256) if n else ())
    none, few, many = (_terms(m, w(n), "mmac_bf16", m.peaks["mmac_bf16"]) for n in (0, 2, 8))
    assert "sync:barrier" not in none
    assert many["sync:barrier"] > few["sync:barrier"] > 0
    assert abs(many["sync:barrier"] / few["sync:barrier"] - 4.0) < 1e-6


def test_fusing_more_handoffs_through_lds_costs_more_barriers():
    from kra.model.capacity import footprint
    from kra.ops.kda import capacity_graph
    g = capacity_graph({"T": 16384, "H": 12}, formulation={"state": "recompute"})
    ops = {o.name: o for o in g.ops}
    one = footprint(g, [ops["wy_bwd_a"]])
    two = footprint(g, [ops["wy_bwd_a"], ops["wy_bwd_b"]])
    assert two["barriers"] > one["barriers"]
