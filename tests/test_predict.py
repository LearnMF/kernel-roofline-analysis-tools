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


def test_chain_needs_machine_step_latency():
    import json
    m = json.loads(BW1000.read_text())
    m.pop("structural")
    r = predict(TheoryMachine(m), "kda", {"T": 16384, "H": 12}, "bf16", "fwd+bwd")
    assert r["bottleneck"] != "serial"
    assert any("serial chain is NOT" in n for n in r["notes"])
