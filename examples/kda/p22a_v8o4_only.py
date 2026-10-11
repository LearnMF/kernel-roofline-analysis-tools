"""kra P-22a (round 5): P-22 restricted to its fwd_h_bn half.  Applies P-22, then restores dhu's
aN(dv2) emit and s4a's original dva loads (the dhu half lost at H >= 48: R21).  Kept: v8o4 MODE 4
(no aN(v_new) copy) + s4b rebuilding avn from the C-form Vn (in-register 4x4 transpose).
Usage: python3 p22a_v8o4_only.py <tree root>"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
subprocess.run([sys.executable, str(Path(__file__).with_name("p22_no_an_copies.py")), str(ROOT)], check=True)


def sub(path, old, new):
    t = path.read_text()
    assert t.count(old) == 1, (path.name, old[:90])
    path.write_text(t.replace(old, new))


WY = ROOT / "csrc/g2/g2_wy.cuh"
K3 = ROOT / "hip_kda/ops/k3_g2.py"
sub(WY, """        v4bh dva[8];
        if (p.dv != nullptr) {
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) dva[vt] = aN(dvN, vt);
        } else {                // kra P-22: from the C-form dV2n via the in-register transpose
            const bf16* dvc = p.dv2c + g2_nat_blk(i, hh, H, p.NT, false, 8192) + r * 8 * 256 + l * 4;
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) dva[vt] = g2_c2a(ld8(dvc + vt * 256), l);
        }
        #pragma unroll""",
    """        v4bh dva[8];
        #pragma unroll
        for (int vt = 0; vt < 8; ++vt) dva[vt] = aN(dvN, vt);
        #pragma unroll""")
sub(K3, """        bN_dh, dV2n, aN_dv2 = g2_ops.dhu_bn(KgA, Qn, Wb, dOT, dVn, Gn, scale,
                                            emit_dv2n=emit_dv2n or no_an, seg=ctx.seg,
                                            emit_an2=not no_an)
        dv2_c = None
        if no_an:
            dv2_c, aN_dv2 = dV2n, torch.empty(0, dtype=dV2n.dtype, device=dV2n.device)""",
    """        # kra P-22a: dhu keeps its aN(dv2) emit (dropping it lost at H >= 48, rule R21)
        bN_dh, dV2n, aN_dv2 = g2_ops.dhu_bn(KgA, Qn, Wb, dOT, dVn, Gn, scale,
                                            emit_dv2n=emit_dv2n, seg=ctx.seg)
        dv2_c = None""")
print("P-22a APPLIED")
