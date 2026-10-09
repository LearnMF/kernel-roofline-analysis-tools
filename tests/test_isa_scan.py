"""Unit tests for the static ISA scans (synthetic gfx9-style assembly)."""
import unittest

from kra.isa.scan import kernel_body, load_use_slack, mathlib_guard

ASM = """
kern:
\ts_load_dwordx2 s[0:1], s[4:5], 0x0
\tglobal_load_dword v1, v[2:3], off
\ts_waitcnt vmcnt(0)
\tv_add_f32_e32 v1, v1, v1
.LBB0_1:
\tglobal_load_dwordx2 v[4:5], v[2:3], off
\tv_mmac_f32_16x16x16_bf16 v[8:11], v[4:5], v[4:5], v[8:11]
\tv_add_u32_e32 v6, 1, v6
\ts_waitcnt vmcnt(0)
\tv_mmac_f32_16x16x16_bf16 v[8:11], v[4:5], v[4:5], v[8:11]
\ts_cbranch_scc1 .LBB0_1
\tv_cmp_gt_f32_e32 vcc, s10, v1
\tv_cndmask_b32_e32 v2, 0, v3, vcc
\tv_add_f32_e32 v1, v1, v2
\tv_exp_f32_e32 v1, v1
\tv_cndmask_b32_e32 v2, 1.0, v4, vcc
\tv_mul_f32_e32 v1, v1, v2
\tv_exp_f32_e32 v7, v7
\ts_endpgm
.Lfunc_end0:
"""


class IsaScanTest(unittest.TestCase):
    def setUp(self):
        self.body = kernel_body(ASM, "kern")

    def test_body_extraction(self):
        self.assertTrue(self.body[0].strip().startswith("s_load_dwordx2"))
        self.assertTrue(self.body[-1].strip() == "s_endpgm")

    def test_load_use_slack_and_hot_loop(self):
        s = load_use_slack(self.body)
        self.assertEqual(len(s), 2)
        pro, loop = s
        self.assertEqual(pro["slack"], 0)
        self.assertFalse(pro["in_hot_loop"])        # prologue: once per CTA (P-5/P-6 lesson)
        self.assertEqual(loop["slack"], 2)          # mmac + add between load and wait
        self.assertTrue(loop["in_hot_loop"])
        self.assertEqual(loop["loop_depth"], 1)

    def test_mathlib_guard(self):
        r = mathlib_guard(self.body)
        self.assertEqual(r["v_exp"], 2)
        self.assertEqual(r["guarded_exp"], 1)       # the bare v_exp_f32 v7 is not guarded


if __name__ == "__main__":
    unittest.main()
