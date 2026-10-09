"""Unit tests for L0 summarization and the ISA check (no GPU needed)."""
import tempfile
import unittest
from pathlib import Path

from kra.calibrate import run as R

ROWS = [
    {"bench": "device", "name": "X", "cu_count": 80, "clock_khz": 1500000,
     "memory_clock_khz": 1800000, "memory_bus_width": 4096, "warp_size": 64},
    {"bench": "hbm_read", "config": {"grid": 320}, "GBps_best": 1300.0, "GBps_med": 1290.0},
    {"bench": "hbm_read", "config": {"grid": 640}, "GBps_best": 1337.6, "GBps_med": 1274.9},
    {"bench": "l2_read", "config": {"resident": True}, "GBps_best": 5900.0, "GBps_med": 5890.0},
    {"bench": "l2_read", "config": {"resident": False}, "GBps_best": 9999.0, "GBps_med": 1.0},
    {"bench": "mmac_bf16", "config": {}, "TOPS_best": 475.956, "TOPS_med": 475.7},
    {"bench": "sfu_exp2", "config": {}, "TOPS_best": 1.916, "TOPS_med": 1.915},
    {"bench": "gmem_latency", "config": {"working_set": 16384}, "cycles_per_step": 156.0, "ns_per_step": 104.0,
     "shader_clock_mhz": 1500.0},
    {"bench": "gmem_latency", "config": {"working_set": 536870912}, "cycles_per_step": 571.9,
     "ns_per_step": 381.2, "shader_clock_mhz": 1500.0},
    {"bench": "barrier", "config": {"block": 256, "mode": "syncthreads"}, "cycles_per_step": 52.0,
     "ns_per_step": 34.7, "shader_clock_mhz": 1500.0},
    {"bench": "launch_stream", "config": {}, "us_per_kernel_best": 1.96, "us_per_kernel_med": 2.26},
]

ASM = """
_Z11k_mmac_bf16ILi8EEviPf:   ; @foo
\tv_mmac_f32_16x16x16_bf16 v[0:3], v[4:5], v[4:5], v[0:3]
\ts_endpgm
_Z12k_mmac_f32x8ILi8EEviPf:
\tv_mmac_16x16x8_f32 v[0:3], v[4:5], v[4:5], v[0:3]
\ts_endpgm
"""


class CalibrateTest(unittest.TestCase):
    def test_summarize(self):
        s = R.summarize(ROWS)
        p = s["peaks"]
        self.assertEqual(p["hbm_read"]["value"], 1337.6)
        self.assertEqual(p["hbm_read"]["config"], {"grid": 640})
        self.assertAlmostEqual(p["hbm_read"]["theoretical"], 1843.2)
        self.assertEqual(p["l2_read"]["value"], 5900.0)          # non-resident run excluded
        self.assertAlmostEqual(p["sfu_exp2"]["value"], 1916.0)   # TOPS -> Gop/s
        self.assertAlmostEqual(p["mmac_bf16"]["ops_per_cu_per_clock"], 3966.3, places=1)
        self.assertAlmostEqual(s["ridge_points_flop_per_byte"]["mmac_bf16_vs_hbm_read"], 355.8, places=1)
        self.assertEqual(s["latency"]["gmem_small_ws"]["cycles"], 156.0)
        self.assertEqual(s["latency"]["gmem_hbm_ws"]["cycles"], 571.9)
        self.assertEqual(s["latency"]["barrier"]["syncthreads_256"]["cycles"], 52.0)
        self.assertEqual(s["launch"]["launch_stream"]["best_us"], 1.96)

    def test_isa_check(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "mb-hip-amdgcn-amd-amdhsa-gfx936.s").write_text(ASM)
            r = R.isa_check(Path(d), "gfx936")
            self.assertTrue(r["kernels"]["k_mmac_bf16"]["ok"])
            self.assertTrue(r["kernels"]["k_mmac_f32x8"]["ok"])      # gfx936 spelling accepted
            self.assertFalse(r["kernels"]["k_exp2"]["ok"])           # absent symbol reported
            self.assertFalse(r["ok"])


if __name__ == "__main__":
    unittest.main()
