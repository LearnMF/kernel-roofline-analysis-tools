"""Unit tests for L2 PMC parsing and the L4 bound model (synthetic inputs, no GPU)."""
import tempfile
import unittest
from pathlib import Path

from kra.model import bounds as B
from kra.pmc import hipprof_csv as P

MACHINE = {
    "id": "test",
    "device": {"cu_count": 80, "clock_khz": 1500000},
    "peaks": {"hbm_read": {"value": 1300.0}, "hbm_write": {"value": 1100.0},
              "mmac_bf16": {"value": 476.0},
              "hbm_read_vs_ctas": {"curve": {"1": 40.0, "48": 1200.0, "80": 1300.0}}},
    "latency": {"lds": {"cycles": 64}, "mmac_bf16_dependent": {"cycles": 44},
                "gmem_hbm_ws": {"cycles": 570}, "gmem_l2_ws": {"cycles": 380},
                "barrier": {"lds_barrier_256": {"cycles": 52}, "lds_barrier_512": {"cycles": 56}}},
    "launch": {"launch_stream": {"best_us": 2.0}},
}


def kern(name, t_us, ctas, block=256, rd=0.0, wr=0.0, slots=0.0, insts=0.0, conflict=0.0):
    return {"name": name, "t_us": t_us, "ctas": ctas, "block": block, "cycles": t_us * 1500,
            "hbm_rd_bytes": rd, "hbm_wr_bytes": wr, "valu_slots": slots, "valu_insts": insts,
            "lds_bank_conflict_cycles": conflict, "arch_vgpr": 64, "accum_vgpr": 0,
            "lds_bytes": 0, "scratch": 0}


class PmcCsvTest(unittest.TestCase):
    def test_bytes_and_join(self):
        rd_csv = ("Index,KernelName,grd,wgr,lds,scr,arch_vgpr,accum_vgpr,sgpr,GRBM_GUI_ACTIVE,"
                  "SQ_ACTIVE_INST_VALU,SQ_INSTS_VALU,TCC_EA_RDREQ[0],TCC_EA_RDREQ[1],TCC_EA_RDREQ_32B[0],"
                  "TCC_EA_RDREQ_32B[1],BeginNs,EndNs\n"
                  "0,spin_kernel,64,64,0,0,8,0,16,100,0,0,0,0,0,0,0,1000\n"
                  "1,kern_a,163840,512,0,0,16,0,32,1500,300,100,10,6,2,0,1000,3000\n")
        wr_csv = ("Index,KernelName,grd,wgr,TCC_EA_WRREQ[0],TCC_EA_WRREQ_64B[0],BeginNs,EndNs\n"
                  "0,spin_kernel,64,64,0,0,0,1000\n"
                  "1,kern_a,163840,512,5,4,10000,12500\n")
        with tempfile.TemporaryDirectory() as d:
            Path(d, "r.csv").write_text(rd_csv)
            Path(d, "w.csv").write_text(wr_csv)
            recs = P.window(P.load(Path(d, "r.csv"), Path(d, "w.csv")), "spin")
        self.assertEqual(len(recs), 1)
        k = recs[0]
        self.assertEqual(k["hbm_rd_bytes"], 64 * (16 - 2) + 32 * 2)    # instances summed
        self.assertEqual(k["hbm_wr_bytes"], 64 * 4 + 32 * 1)
        self.assertEqual(k["ctas"], 320)
        self.assertAlmostEqual(k["t_us"], 2.0)                          # min of the two replays
        self.assertEqual(k["valu_slots"], 300)


class BoundsTest(unittest.TestCase):
    def setUp(self):
        self.M = B.Machine(MACHINE)

    def test_bw_curve(self):
        self.assertAlmostEqual(self.M.bw, 1300e9)                    # max of measured peaks
        self.assertAlmostEqual(self.M.bw_for(48), 1200e9)
        self.assertAlmostEqual(self.M.bw_for(24), 40e9 + (1200e9 - 40e9) * 23 / 47)
        self.assertAlmostEqual(self.M.bw_for(200), 1300e9)

    def test_chain(self):
        self.assertEqual(self.M.chain_cycles({"barrier": 2, "lds": 2, "mmac_dep": 12}, 256),
                         2 * 52 + 2 * 64 + 12 * 44)
        with self.assertRaises(ValueError):
            self.M.chain_cycles({"warp_shuffle": 1}, 256)

    def test_memory_bound_kernel_classified_B(self):
        th = B.thresholds()
        k = kern("stream", t_us=1000.0, ctas=640, rd=1.2e9, wr=0.0)    # 1.2 GB/ms = 92% of 1300
        m = B.kernel_metrics(k, self.M, th)
        c = B.classify(m, False, th, 80)
        self.assertEqual(c["class"], "B.memory.hbm")
        self.assertEqual(c["regime"], "near-limit")

    def test_small_grid_serial_is_C(self):
        th = B.thresholds()
        # 400 us = 600k cycles x 80 CUs = 48M CU-cycles; 5M conflict cycles = 10.4%
        k = kern("rec", t_us=400.0, ctas=48, rd=50e6, wr=30e6, slots=4e6, insts=3e6, conflict=5e6)
        m = B.kernel_metrics(k, self.M, th)
        c = B.classify(m, True, th, 80)
        self.assertEqual(c["class"], "C")
        self.assertEqual(c["subtypes"], ["C.parallelism", "C.serial"])
        self.assertAlmostEqual(m["mmac_slot_share"], 0.5)
        self.assertIn("D.lds_conflict", [x["id"] for x in c["causes"]])

    def test_analyze_end_to_end(self):
        spec = {"operator": "toy", "vars": {"NT": "T // 64"},
                "groups": [{"launcher": "prep", "kernels": ["prep_a", "prep_b?"]},
                           {"launcher": "rec", "kernels": ["rec_k"],
                            "serial": {"steps": "NT", "chain": {"barrier": 2, "lds": 2, "mmac_dep": 12},
                                       "chain_min": {"lds": 1, "mmac_dep": 2}}}]}
        pmc = {"kernels": [kern("prep_a", 100.0, 640, rd=60e6, wr=60e6),
                           kern("torch_copy", 5.0, 80, rd=1e6, wr=1e6),
                           kern("rec_k", 400.0, 48, rd=50e6, wr=30e6, slots=4e6, insts=3e6)]}
        optrace = {"calls": [{"launcher": "prep", "in_bytes": 50e6, "out_bytes": 40e6},
                             {"launcher": "helper", "in_bytes": 0, "out_bytes": 0},
                             {"launcher": "rec", "in_bytes": 40e6, "out_bytes": 20e6}]}
        res = B.analyze(MACHINE, pmc, spec, optrace, {"T": 8192})
        names = [r["launcher"] for r in res["rows"]]
        self.assertEqual(names, ["prep", "(unattributed)", "rec"])  # optional prep_b skipped
        rec = res["rows"][2]
        self.assertEqual(rec["critical_path"]["steps"], 128)
        self.assertAlmostEqual(rec["critical_path"]["T_cp_min"], 128 * (64 + 88) / 1.5e9 * 1e6)
        # prior bound: interface bytes at the 48-CTA bandwidth dominates the chain here
        exp = (40e6 + 20e6) / 1200e9 * 1e6
        self.assertAlmostEqual(rec["components_us"]["par_mem_min"], exp)
        self.assertEqual(rec["binding_prior"], "par_mem_min")
        self.assertAlmostEqual(rec["T_lower_prior_us"], exp)
        self.assertEqual(rec["verdict"], "large_headroom")
        self.assertAlmostEqual(res["totals"]["t_us"], 505.0)
        self.assertIsNone(res["fusion_floor"])          # optrace without ptrs
        self.assertEqual(res["audit"], [])

    def test_fusion_floor(self):
        groups = {"a": {"phase": "fwd"}, "b": {"phase": "fwd"}, "c": {"phase": "bwd"}}
        calls = [
            {"launcher": "a", "inputs": [{"ptr": 1, "bytes": 100}], "outputs": [{"ptr": 2, "bytes": 50}]},
            {"launcher": "b", "inputs": [{"ptr": 2, "bytes": 50}], "outputs": [{"ptr": 3, "bytes": 30}]},
            {"launcher": "c", "inputs": [{"ptr": 3, "bytes": 30}, {"ptr": 4, "bytes": 10}],
             "outputs": [{"ptr": 5, "bytes": 20}]},
        ]
        ff = B.fusion_floor(calls, groups, self.M)
        self.assertEqual(ff["fwd"]["in_MB"] * 1e6, 100)      # tensor 2 stays inside fwd
        self.assertEqual(ff["fwd"]["out_MB"] * 1e6, 30)      # tensor 3 leaves fwd (saved for bwd)
        self.assertEqual(ff["bwd"]["in_MB"] * 1e6, 40)
        self.assertEqual(ff["bwd"]["out_MB"] * 1e6, 20)

    def test_bound_violation_is_audited(self):
        th = B.thresholds()
        self.assertEqual(B.verdict(1.05, 1.0, "B.memory.hbm", th), "bound_violated")
        self.assertEqual(B.verdict(0.6, 0.7, "B.compute.valu", th), "throughput_bound")


if __name__ == "__main__":
    unittest.main()
