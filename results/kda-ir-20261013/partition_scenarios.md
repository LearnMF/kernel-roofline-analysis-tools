core LDS per op (KB): current / lean, C=64 and C=32 (measured at C=64 in parentheses)
  re_gate   C=64   2.0 /   2.0  C=32   1.0 /   1.0   (measured 2.0)
  re_wy     C=64  26.2 /  17.8  C=32  11.9 /   9.6   (measured 26.2)
  dav       C=64  17.0 /  17.0  C=32   9.0 /   9.0   (measured 17.0)
  wy_bwd_a  C=64  29.8 /  25.8  C=32  13.4 /  11.4   (measured 29.8)
  wy_bwd_b  C=64  32.0 /  24.0  C=32  24.4 /  16.4   (measured 32.0)
  tail      C=64   6.0 /   6.0  C=32   3.0 /   3.0   (measured 6.0)

| fused group | handoff saved B/tok*head | LDS KB (res): current C=64 | lean C=64 | current C=32 | lean C=32 |
|---|---:|---:|---:|---:|---:|
| re_gate+re_wy | 512 | 58.2 (1) | 49.8 (1) | 27.9 (2) | 25.6 (2) |
| wy_bwd_a+wy_bwd_b | 1024 | 64.0 (1) | 57.8 (1) | 36.4 (1) | 28.4 (2) |
| dav_a+wy_bwd_a+wy_bwd_b | 1536 | 80.0 (0) | 73.8 (0) | 40.4 (1) | 32.4 (1) |
| wy_bwd_b+tail | 3592 | 104.0 (0) | 96.0 (0) | 60.4 (1) | 52.4 (1) |
| dav_a+wy_bwd_a+wy_bwd_b+tail | 5136 | 152.0 (0) | 144.0 (0) | 76.4 (0) | 68.4 (0) |

formulation crossing (scan <-> chunk, W+R): C=64 8224, C=32 10176 B/tok*head -> going to C=32 costs +1952 B/tok*head before any fusion saving
