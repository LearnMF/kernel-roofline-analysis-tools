"""kra P-87 (bitwise, Python only): the saved-h route now runs up to H = 48 at every length.

With P-84c (wu's no-Kn phase 3 no longer re-reads q + fp32 g) the saved-h backward got cheaper; re-measured on bw52
(results/kda-gfx936-20261016-p83-p85 §P-87, pipe_cfg, all outputs bit-identical), F+B vs the P-82b policy:
H=48: 4K 1.041, 8K 1.027, 16K 1.052 (32K+ already on); H=32: 8K 1.010, 16K 1.008; H=96 mixed (4K 1.000, 8K 1.055,
16K 0.984) -> stays off.  KDA_G2_SAVE_H_MAX_H default 24 -> 48 (memory: the approved +512 B/token*head, = FLA default).
    python3 p87_save_h_48.py <tree>"""
import sys
from pathlib import Path

k3 = Path(sys.argv[1]) / "hip_kda" / "ops" / "k3_g2.py"
s = k3.read_text()
old = """                     and (H <= int(os.environ.get("KDA_G2_SAVE_H_MAX_H", "24"))"""
new = """                     and (H <= int(os.environ.get("KDA_G2_SAVE_H_MAX_H", "48"))   # kra P-87: 24 -> 48"""
assert s.count(old) == 1
k3.write_text(s.replace(old, new))
print("P-87 APPLIED")
