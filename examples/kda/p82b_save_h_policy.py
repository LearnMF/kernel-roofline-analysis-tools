"""kra P-82b (bitwise, on top of P-82): where the saved-h route runs.
  * extent (pipe_cfg, results/kda-gfx936-20261016-p82-save-h §P-82b): F+B with the h save vs the recompute --
    16K/24 1.029, 16K/48 1.003, 32K/48 1.017 (wide v8o4w MODE 5), 8K/96 0.983 -> the route runs at H <= 24, and at
    40 <= H <= 48 when the wide form runs (H >= KDA_G2_V8O4W_MIN_H = 40, NT >= KDA_G2_V8O4W_MIN_NT = 512):
        KDA_G2_SAVE_H_MAX_H (24), KDA_G2_SAVE_H_WIDE_MAX_H (48)
  * keep-state: where the h route runs it replaces the full B2 hand-over (MODE 2: bN + Vn + aN, 1024 B/token*head)
    -- half the bytes and faster (16K/12: h route -255 us vs keep-state -160 us), bit-identical either way.
    python3 p82b_save_h_policy.py <tree>"""
import sys
from pathlib import Path

k3 = Path(sys.argv[1]) / "hip_kda" / "ops" / "k3_g2.py"
s = k3.read_text()
old = """            keep = keep_state(ctx)
            # kra P-82: save h (bN) whenever a backward follows -> the backward drops its recompute
            # (H <= KDA_G2_SAVE_H_MAX_H, default 20 = the kp/BP instances; at H = 48 / 96 the forward bN emit
            # costs +51% of the recurrence and the save does not pay)
            hsave = (not keep and any(ctx.needs_input_grad)
                     and os.environ.get("KDA_G2_SAVE_H", "1").strip() != "0"
                     and H <= int(os.environ.get("KDA_G2_SAVE_H_MAX_H", "20")))"""
new = """            # kra P-82: save h (bN) whenever a backward follows -> the backward drops its recompute.
            # kra P-82b: at H <= KDA_G2_SAVE_H_MAX_H (24), or H <= KDA_G2_SAVE_H_WIDE_MAX_H (48) on the wide
            # recurrence (NT >= KDA_G2_V8O4W_MIN_NT); there it also replaces the keep-state B2 hand-over.
            hsave = (any(ctx.needs_input_grad) and os.environ.get("KDA_G2_SAVE_H", "1").strip() != "0"
                     and (H <= int(os.environ.get("KDA_G2_SAVE_H_MAX_H", "24"))
                          or (int(os.environ.get("KDA_G2_V8O4W_MIN_H", "40")) <= H
                              <= int(os.environ.get("KDA_G2_SAVE_H_WIDE_MAX_H", "48"))
                              and NT >= int(os.environ.get("KDA_G2_V8O4W_MIN_NT", "512")))))
            keep = keep_state(ctx) and not hsave"""
assert s.count(old) == 1
s = s.replace(old, new)
k3.write_text(s)
print("P-82b APPLIED")
