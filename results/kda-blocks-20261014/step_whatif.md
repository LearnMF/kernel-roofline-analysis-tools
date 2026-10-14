
fwd MODE 1: baseline 3333 cycles (2.222 us/step)
  no 64-bit address arithmetic (SGPR bases / buffer offsets)   3228   -3.1%
  ds_write_b16 x4 -> ds_write_b64 (+2 perm)                    3079   -7.6%
  half the SALU (2-step unroll)                                3333   +0.0%
  no s_nop                                                     3332   -0.1%
  all four                                                     2970  -10.9%

recompute MODE 4: baseline 3015 cycles (2.010 us/step)
  no 64-bit address arithmetic (SGPR bases / buffer offsets)   2743   -9.0%
  ds_write_b16 x4 -> ds_write_b64 (+2 perm)                    2870   -4.8%
  half the SALU (2-step unroll)                                3007   -0.3%
  no s_nop                                                     3012   -0.1%
  all four                                                     2558  -15.1%

dhu 8-wave: baseline 3194 cycles (2.130 us/step)
  no 64-bit address arithmetic (SGPR bases / buffer offsets)   3192   -0.1%
  ds_write_b16 x4 -> ds_write_b64 (+2 perm)                    3196   +0.1%
  half the SALU (2-step unroll)                                3200   +0.2%
  no s_nop                                                     3192   -0.1%
  all four                                                     3192   -0.1%
