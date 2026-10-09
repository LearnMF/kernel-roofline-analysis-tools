"""Unit tests for the L1 timeline analyzer on a synthetic hipprof-format trace."""
import json
import tempfile
import unittest
from pathlib import Path

from kra.timeline import analyze as A
from kra.timeline.hipprof_json import load

T0 = 1_000_000_000_000  # absolute ns base


def _op(name, b, e, idx, cat="HIPOPS", tid="Stream0", queue="0"):
    return {"ph": "X", "cat": cat, "name": name, "pid": 3 if cat == "HIPOPS" else 2, "tid": tid,
            "ts": (T0 + b) / 1000 - T0 / 1000, "dur": (e - b) / 1000,
            "args": {"BeginNs": str(T0 + b), "EndNs": str(T0 + e), "devId": 0, "queueId": queue,
                     "name": name, "index": idx}}


def _api(name, b, e):
    return {"ph": "X", "cat": "HIP", "name": name, "pid": 1, "tid": "Thread1",
            "ts": b / 1000, "dur": (e - b) / 1000, "args": {"BeginNs": str(T0 + b), "EndNs": str(T0 + e)}}


def _flow(fid, submit_ns, op_ts_ns, op_pid=3):
    return [{"ph": "s", "cat": "DataFlow", "id": fid, "pid": 1, "tid": "Thread1", "name": "dep",
             "ts": submit_ns / 1000},
            {"ph": "t", "cat": "DataFlow", "id": fid, "pid": op_pid, "tid": "Stream0", "name": "dep",
             "ts": op_ts_ns / 1000 + 1e-4}]


def build_trace():
    """marker | k1 [0,100us] | gap 10us (host late) | k2 [110,200] | k3 [200,300] queued |
    gap 50us with hipDeviceSynchronize | k4 [350,400] (submitted early)"""
    us = 1000
    ev = [_op("at::cuda::spin_kernel(long)", -500 * us, -10 * us, 0)]
    ev += [_op("void ns::k1<int>(P)", 0, 100 * us, 1), _op("k2(P)", 110 * us, 200 * us, 2),
           _op("void ns::(anonymous namespace)::k3(short const*, int)", 200 * us, 300 * us, 3),
           _op("k4", 350 * us, 400 * us, 4)]
    ev += _flow(11, -5 * us, 0) + _flow(12, 105 * us, 110 * us) + _flow(13, 50 * us, 200 * us) \
        + _flow(14, 90 * us, 350 * us)
    ev += [_api("hipLaunchKernel", 104 * us, 106 * us), _api("hipDeviceSynchronize", 300 * us, 340 * us)]
    return {"traceEvents": ev}


class TimelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "t.json"
        self.path.write_text(json.dumps(build_trace()))

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_and_flows(self):
        tr = load(self.path)
        self.assertEqual(len(tr.ops), 5)
        k2 = next(o for o in tr.ops if o.name.startswith("k2"))
        self.assertEqual(k2.submit_ns - (10**12), 105_000)
        self.assertEqual(k2.stream, "q0")

    def test_window_metrics_and_gap_causes(self):
        machine = {"launch": {"launch_stream": {"best_us": 2.0}, "launch_graph": {"best_us": 1.5}}}
        res = A.analyze(load(self.path), marker="spin_kernel", machine=machine)
        self.assertEqual(res["n_windows"], 1)
        w = res["representative"]
        self.assertAlmostEqual(w["span_us"], 400.0)
        self.assertAlmostEqual(w["busy_us"], 340.0)
        self.assertAlmostEqual(w["bubble_ratio"], 60 / 400)
        causes = {g["before"]: g["cause"] for g in w["largest_gaps"]}
        self.assertEqual(causes["k2"], "host_late")
        self.assertEqual(causes["k4"], "sync")
        self.assertEqual(w["n_queued_back_to_back"], 1)          # k3 behind k2
        self.assertAlmostEqual(w["embedded_dispatch_us_upper"], 2.0)
        self.assertAlmostEqual(w["recoverable_bubble_us_upper"], 10 - 1.5)
        self.assertTrue(w["verdict"]["system_bound"])
        self.assertIn("k3", w["per_kernel"])                      # anonymous-namespace name parsed

    def test_short_name(self):
        self.assertEqual(A.short_name("void g2::r4::g2_fwd_prep_a_kernel<1, false>(g2::r4::PrepAParams)"),
                         "g2_fwd_prep_a_kernel")
        self.assertEqual(A.short_name("g2::(anonymous namespace)::g2_l2n_apply_kernel(short const*, int)"),
                         "g2_l2n_apply_kernel")

    def test_markdown_renders(self):
        res = A.analyze(load(self.path), marker="spin_kernel", reference_ms=0.35)
        md = A.to_markdown(res)
        self.assertIn("Trust check", md)
        self.assertIn("host_late", md)


if __name__ == "__main__":
    unittest.main()
