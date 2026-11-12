// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// kra P-938s3: the 3-CTA s4b instance (gfx938) in its own translation unit -- sharing a code object with the default
// s4b kernel measured 3-4% slower for identical machine code.  Only the s4b section of g2_wy.cuh is compiled here.
#define G2_S4B3_TU 1
#include "g2/g2_common.h"
#include <type_traits>
#include "g2/g2_wy.cuh"

namespace g2 {
int g2_s4b3_launch(const WyS4BParams& p, dim3 grid, bool vl, hipStream_t s) {
    if (vl) hipLaunchKernelGGL((p3_wy_s4b_kernel<true, true>), grid, dim3(256), 0, s, p);
    else hipLaunchKernelGGL((p3_wy_s4b_kernel<false, true>), grid, dim3(256), 0, s, p);
    return (int)hipGetLastError();
}
}  // namespace g2
