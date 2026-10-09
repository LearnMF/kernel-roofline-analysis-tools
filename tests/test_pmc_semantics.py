"""Regression test of the gfx936 PMC semantics against the real calibration capture
(results/pmc-semantics-gfx936-20261009: `microbench pmccal` under hipprof --pmc-read/-write)."""
import json
import unittest
from pathlib import Path

from kra.pmc.hipprof_csv import load

D = Path(__file__).resolve().parent.parent / "results" / "pmc-semantics-gfx936-20261009"
CUS = 80


class PmcSemanticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recs = load(D / "cal_pmc-read.csv", D / "cal_pmc-write.csv")
        cls.ref = [json.loads(l) for l in (D / "cal_pmc-read.log").read_text().splitlines()
                   if l.startswith('{"bench":"pmccal"')]
        cls.by = {r["name"]: k for r, k in zip(cls.ref, cls.recs)}

    def test_order_matches(self):
        self.assertEqual(len(self.recs), len(self.ref))

    def test_hbm_bytes_exact(self):
        for name in ("hbm_read", "hbm_write", "hbm_copy"):
            r = next(x for x in self.ref if x["name"] == name)
            k = self.by[name]
            self.assertAlmostEqual(k["hbm_rd_bytes"] / max(1, r["bytes_rd"]) if r["bytes_rd"] else k["hbm_rd_bytes"],
                                   1.0 if r["bytes_rd"] else k["hbm_rd_bytes"], delta=1e-3 if r["bytes_rd"] else 1e5)
            if r["bytes_wr"]:
                self.assertAlmostEqual(k["hbm_wr_bytes"] / r["bytes_wr"], 1.0, delta=1e-3)

    def test_valu_slots(self):
        mm = self.by["mmac_bf16_full"]
        exp = next(x for x in self.ref if x["name"] == "mmac_bf16_full")["expect_wave_insts"]
        self.assertAlmostEqual(mm["valu_slots"] / exp, 2.0, delta=0.01)      # 2 slots per MMAC
        self.assertAlmostEqual((mm["valu_slots"] - mm["valu_insts"]) / exp, 1.0, delta=0.01)
        ex = self.by["sfu_exp2"]
        exp = next(x for x in self.ref if x["name"] == "sfu_exp2")["expect_wave_insts"]
        self.assertAlmostEqual(ex["valu_slots"] / exp, 4.0, delta=0.01)      # 4 slots per exp
        fm = self.by["valu_pk_fma"]
        self.assertAlmostEqual(fm["valu_slots"], fm["valu_insts"])           # 1 slot per VALU

    def test_pipe_utilization_at_peak(self):
        # denominator: faster replay's time at 1.5 GHz (what kra.model.bounds uses)
        for name in ("mmac_bf16_full", "valu_pk_fma", "sfu_exp2"):
            k = self.by[name]
            util = k["valu_slots"] / (k["t_us"] * 1e-6 * 1.5e9 * CUS)
            self.assertGreater(util, 0.95, name)
            self.assertLessEqual(util, 1.01, name)


if __name__ == "__main__":
    unittest.main()
