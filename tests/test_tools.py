"""kra.tools 通用工具的回归测试（不需要 HCU：门禁、交替计时、Roofline 计算与画图、ISA 解析）。"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kra.tools import abba, gate, isa, roofline

ROOT = Path(__file__).resolve().parent.parent


class GateTest(unittest.TestCase):
    def test_bitwise(self):
        a = {"o": np.arange(6, dtype=np.float32), "dq": np.ones(3, np.float32)}
        b = {k: v.copy() for k, v in a.items()}
        self.assertTrue(gate.bitwise(a, b)[0])
        b["dq"][1] = np.nextafter(np.float32(1), np.float32(2))     # 只差 1 ulp
        ok, rows = gate.bitwise(a, b)
        self.assertFalse(ok)
        self.assertEqual([r["status"] for r in rows if r["tensor"] == "dq"], ["differ"])
        self.assertFalse(gate.bitwise(a, {"o": a["o"]})[0])            # 缺张量

    def test_ref64_and_cli(self):
        rng = np.random.default_rng(0)
        ref = {"o": rng.standard_normal(1000)}
        base = {"o": ref["o"] + 1e-3 * rng.standard_normal(1000)}
        good = {"o": ref["o"] + 0.9e-3 * rng.standard_normal(1000)}
        bad = {"o": ref["o"] + 3e-3 * rng.standard_normal(1000)}
        self.assertFalse(gate.ref64(ref, base, bad)[0])
        with tempfile.TemporaryDirectory() as d:
            for n, x in (("ref", ref), ("base", base), ("good", good)):
                gate.save(f"{d}/{n}.npz", x)
            rc = gate.main(["ref64", f"{d}/ref.npz", f"{d}/base.npz", f"{d}/good.npz", "--max-ratio", "1.2"])
            self.assertEqual(rc, 0)
            self.assertEqual(gate.main(["bitwise", f"{d}/base.npz", f"{d}/base.npz"]), 0)


class AbbaTest(unittest.TestCase):
    def test_interleaved_order_and_ratio(self):
        py = sys.executable
        a = f'{py} -c "import json;print(json.dumps({{\'t\': 2.0}}))"'
        b = f'{py} -c "import json;print(\'noise\');print(json.dumps({{\'t\': 1.0}}))"'
        rows, s = abba.abba(a, b, ["t"], rounds=2)
        self.assertEqual([r["tag"] for r in rows], list("ABBAABBA"))
        self.assertAlmostEqual(s["t"]["ratio_A_over_B"], 2.0)
        self.assertEqual(s["t"]["n_per_side"], 4)

    def test_idle_wait(self):
        py = sys.executable
        a = f'{py} -c "print(\'{{\\"t\\": 1}}\')"'
        rows, _ = abba.abba(a, a, ["t"], rounds=1, idle_cmd="echo 0")
        self.assertTrue(all(r["wait_s"] < 1 for r in rows))


class RooflineTest(unittest.TestCase):
    def setUp(self):
        self.m = json.load(open(ROOT / "machines" / "gfx936-bw1000.json"))

    def test_kda_example_from_chapter3(self):
        # KDA 16K/TP8：99.1 GFLOP、4.722 GB、5.32 ms -> I≈21、18.6 TFLOPS、MBU≈66%（教程第 3.6 节）
        r = roofline.analyze(self.m, [{"name": "kda", "flops": "99.1e9", "time_us": "5320", "hbm_bytes": "4.722e9"}])[0]
        self.assertAlmostEqual(r["I_HBM"], 21.0, delta=0.1)
        self.assertAlmostEqual(r["TFLOPS"], 18.6, delta=0.1)
        self.assertAlmostEqual(r["u_HBM"], 0.663, delta=0.005)
        self.assertAlmostEqual(r["to_roof"], r["u_HBM"], delta=1e-9)

    def test_hierarchical_plot(self):
        rows = [{"name": "wy_s4b", "flops": "12.1e9", "time_us": "660", "hbm_bytes": "484e6",
                 "l2_bytes": "502e6", "lds_bytes": "2.70e9", "roof": ""}]
        r = roofline.analyze(self.m, rows)[0]
        self.assertAlmostEqual(r["u_LDS"], 0.27, delta=0.01)
        self.assertAlmostEqual(r["u_L2"], 0.125, delta=0.01)
        with tempfile.TemporaryDirectory() as d:
            roofline.plot(self.m, rows, [r], f"{d}/h.png", None)
            self.assertGreater(Path(f"{d}/h.png").stat().st_size, 10000)


class IsaTest(unittest.TestCase):
    def test_real_gfx936_asm(self):
        k = {x["kernel"]: x for x in isa.parse((ROOT / "results/kda-ceiling-cards-20261013/raw/g2_ops_hip.s").read_text())}
        d = k["g2_dhu_w8b_kernel"]
        self.assertEqual((d["vgpr"], d["lds"], d["scratch"]), (116, 20992, 0))
        self.assertGreater(d["_ops"]["v_mmac_f32_16x16x16_bf16"], 0)
        self.assertGreater(len(k), 50)


if __name__ == "__main__":
    unittest.main()
